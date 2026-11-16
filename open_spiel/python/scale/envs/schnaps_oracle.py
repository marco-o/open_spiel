# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The schnaps oracle: an exact solver that sees every hand (a cheating bot).

It measures a ceiling: how much a player could gain over a fair bot if it
knew all four hands. It is not a fair player and is not meant for training.

Pure Python, no JAX. Three parts:

  * Model: OpenSpiel's schnaps rules (games/schnaps/schnaps.cc) re-written
    in Python, one deal, the same action ids. It also builds a seat's
    InfoState (envs.schnaps INFOSTATE), so a JAX rule bot such as Elena can
    play alongside. The tests check it move for move against pyspiel.
  * PlaySolver: alpha-beta minimax with a transposition table over the card
    play, from full information: the oracle's partnership maximises its
    return, the other minimises it (assumes the opponents also play
    perfectly).
  * Oracle: the bidding solved the same way over a small tree, each
    contract's play value from PlaySolver (flecks only scale a value, so
    each contract and declarer is solved once per deal).

What the oracle does not know: the trump suit is chosen after only three
cards are dealt, before the other hands exist, so the oracle delegates the
trump choice (Oracle.decide returns None there; the caller plays its fair
bot's choice).

Values are in game points for partnership {0, 2} (seats 0 and 2): + when it
wins. A Normal game is worth 1, 2 or 3, a contract its base points, each
times 2^flecks.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

N_PLAYER = 4
N_SUIT = 4
N_RANK = 5
N_CARD = 20
HAND_SIZE = 5
CALLER_CARDS = 3
RANK_POINTS = (2, 3, 4, 10, 11)
CONTRACT_POINTS = (3, 6, 9, 12, 18, 24)
NORMAL, SCHNAPSER, LOMT, BAUERN, JODLER, HERMORT = range(6)
OBER, KOENIG = 1, 2
FIRST_DECLARE = 20
FIRST_TRUMP = 28
PASS = 32
FLECK = 33
FIRST_CALL = 34
WINNING_POINTS = 66
#: The phases, as the JAX engine numbers them (jax.games.schnaps PHASE_*).
DEAL_CALLER, CHOOSE_TRUMP, DEAL_REST, BIDDING, PLAY, OVER = range(6)
INF = 10**6


def suit_of(card: int) -> int:
  return card // N_RANK


def rank_of(card: int) -> int:
  return card % N_RANK


def team_of(seat: int) -> int:
  return seat % 2


def max_bids(max_flecks: int) -> int:
  """The most bids one bidding can take (jax.games.schnaps.max_bids)."""
  return 3 * (5 + 6 * max_flecks) + 4


def cards_of(mask: int) -> list[int]:
  """The cards of a bitmask, ascending."""
  return [c for c in range(N_CARD) if mask >> c & 1]


# ------------------------------- Card play ---------------------------------
#
# A play state is a tuple (hands, trick, leader, score, tricks, pending, n):
#   hands    4 bitmasks of the cards each seat still holds
#   trick    the cards of the trick in progress, in order (seat leader + i)
#   leader   the seat that led it
#   score    (team 0, team 1) points: cards and valid declarations
#   tricks   (team 0, team 1) tricks won
#   pending  (team 0, team 1) declarations awaiting the team's first trick
#   n        tricks played
# The rules of a deal's play are fixed: Rules.


@dataclasses.dataclass(frozen=True)
class Rules:
  """What stays fixed during a deal's play."""

  trump: int  # the trump suit (still set in a Lomt, where it does not count)
  contract: int
  declarer: int

  @property
  def trump_active(self) -> bool:
    return self.contract != LOMT


def initial_play(hands: tuple[int, ...], rules: Rules) -> tuple:
  """The play state when the bidding ends."""
  leader = rules.declarer if rules.contract == LOMT else 0
  return (tuple(hands), (), leader, (0, 0), (0, 0), (0, 0), 0)


def to_move(p: tuple) -> int:
  return (p[2] + len(p[1])) % N_PLAYER


def play_legal(rules: Rules, p: tuple) -> list[int]:
  """The legal play actions (SchnapsState::PlayActions)."""
  hands, trick, leader = p[0], p[1], p[2]
  seat = (leader + len(trick)) % N_PLAYER
  hand = cards_of(hands[seat])
  if not trick:
    actions = list(hand)
    if rules.contract in (NORMAL, SCHNAPSER):
      for s in range(N_SUIT):
        k, o = s * N_RANK + KOENIG, s * N_RANK + OBER
        if hands[seat] >> k & 1 and hands[seat] >> o & 1:
          actions += [FIRST_DECLARE + 2 * s, FIRST_DECLARE + 2 * s + 1]
    return actions
  active = rules.trump_active
  led = suit_of(trick[0])
  best_led, best_trump, trumped = -1, -1, False
  for c in trick:
    s, r = suit_of(c), rank_of(c)
    if active and s == rules.trump:
      trumped = True
      best_trump = max(best_trump, r)
    if s == led:
      best_led = max(best_led, r)
  follow = [c for c in hand if suit_of(c) == led]
  if follow:
    if not (active and trumped and led != rules.trump):
      higher = [c for c in follow if rank_of(c) > best_led]
      if higher:
        return higher
    return follow
  if active:
    trumps = [c for c in hand if suit_of(c) == rules.trump]
    if trumps:
      higher = [c for c in trumps if rank_of(c) > best_trump]
      return higher or trumps
  return hand


def _finish_normal(winner: int, score, tricks) -> int:
  loser = 1 - winner
  if tricks[loser] == 0:
    base = 3
  elif score[loser] < 33:
    base = 2
  else:
    base = 1
  return base if winner == 0 else -base


def _contract_end(rules: Rules, winner_team: int) -> int:
  v = CONTRACT_POINTS[rules.contract]
  return v if winner_team == 0 else -v


def _check_end(rules, score, tricks, n, last_winner, after_trick):
  """The deal's value for team 0 if it ended, else None (CheckGameEnd)."""
  c = rules.contract
  if c == NORMAL:
    for t in (0, 1):
      if score[t] >= WINNING_POINTS:
        return _finish_normal(t, score, tricks)
    if after_trick and n == HAND_SIZE:
      return _finish_normal(team_of(last_winner), score, tricks)
    return None
  if c == SCHNAPSER:
    t = team_of(rules.declarer)
    if score[t] >= WINNING_POINTS:
      return _contract_end(rules, t)
    # A trick won by the other side, or the third trick, ends it.
    if after_trick and (team_of(last_winner) != t or n == 3):
      return _contract_end(rules, 1 - t)
    return None
  if c in (LOMT, BAUERN):
    if not after_trick:
      return None
    t = team_of(rules.declarer)
    if last_winner != rules.declarer:
      return _contract_end(rules, 1 - t)
    if n == HAND_SIZE:
      return _contract_end(rules, t)
    return None
  return None


def _beats(rules: Rules, a: int, b: int) -> bool:
  ta = rules.trump_active and suit_of(a) == rules.trump
  tb = rules.trump_active and suit_of(b) == rules.trump
  if ta != tb:
    return ta
  return suit_of(a) == suit_of(b) and rank_of(a) > rank_of(b)


def play_apply(rules: Rules, p: tuple, action: int):
  """Applies a play action: (the next play state, its value or None).

  The value is the deal's value for team 0 when the action ends the deal.
  """
  hands, trick, leader, score, tricks, pending, n = p
  seat = (leader + len(trick)) % N_PLAYER
  team = team_of(seat)
  if FIRST_DECLARE <= action < FIRST_TRUMP:
    idx = action - FIRST_DECLARE
    s = idx // 2
    card = s * N_RANK + (KOENIG if idx % 2 == 0 else OBER)
    pts = 40 if s == rules.trump else 20
    if tricks[team] > 0:
      score = _add(score, team, pts)
    else:
      pending = _add(pending, team, pts)
    v = _check_end(rules, score, tricks, n, -1, False)
    if v is not None:
      return (hands, trick, leader, score, tricks, pending, n), v
  else:
    card = action
  hands = tuple(h & ~(1 << card) if i == seat else h for i, h in enumerate(hands))
  trick = trick + (card,)
  if len(trick) < N_PLAYER:
    return (hands, trick, leader, score, tricks, pending, n), None
  win = 0
  pts = 0
  for i, c in enumerate(trick):
    pts += RANK_POINTS[rank_of(c)]
    if i > 0 and _beats(rules, c, trick[win]):
      win = i
  winner = (leader + win) % N_PLAYER
  wt = team_of(winner)
  score = _add(score, wt, pts + pending[wt])
  pending = _add(pending, wt, -pending[wt])
  tricks = _add(tricks, wt, 1)
  n += 1
  q = (hands, (), winner, score, tricks, pending, n)
  return q, _check_end(rules, score, tricks, n, winner, True)


def _add(pair, i, v):
  return (pair[0] + v, pair[1]) if i == 0 else (pair[0], pair[1] + v)


class PlaySolver:
  """Exact minimax of a deal's card play, full information (module doc)."""

  def __init__(self, rules: Rules):
    self.rules = rules
    self.table: dict[tuple, tuple[int, int]] = {}
    self.nodes = 0

  def value(self, p: tuple) -> int:
    """The play state's value for team 0 under perfect play by all."""
    return self._search(p, -INF, INF)

  def action_values(self, p: tuple) -> dict[int, int]:
    """Each legal action's exact value for team 0."""
    out = {}
    for a in play_legal(self.rules, p):
      q, v = play_apply(self.rules, p, a)
      out[a] = v if v is not None else self.value(q)
    return out

  def _search(self, p, alpha, beta):
    self.nodes += 1
    lo, hi = self.table.get(p, (-INF, INF))
    if lo >= beta:
      return lo
    if hi <= alpha:
      return hi
    if lo == hi:
      return lo
    alpha, beta = max(alpha, lo), min(beta, hi)
    a0, b0 = alpha, beta
    maximise = team_of(to_move(p)) == 0
    best = -INF if maximise else INF
    for a in play_legal(self.rules, p):
      q, v = play_apply(self.rules, p, a)
      if v is None:
        v = self._search(q, alpha, beta)
      if maximise:
        if v > best:
          best = v
          alpha = max(alpha, v)
      elif v < best:
        best = v
        beta = min(beta, v)
      if alpha >= beta:
        break
    if best <= a0:
      self.table[p] = (lo, min(hi, best))
    elif best >= b0:
      self.table[p] = (max(lo, best), hi)
    else:
      self.table[p] = (best, best)
    return best


# --------------------------------- A deal ----------------------------------


class Model:
  """One deal of OpenSpiel's schnaps, in Python (module doc)."""

  def __init__(self, max_flecks: int = 2):
    self.max_flecks = max_flecks
    self.phase = DEAL_CALLER
    self.holder = [-1] * N_CARD
    self.first_cards: list[int] = []
    self.n_dealt = 0
    self.trump = -1
    self.contract = NORMAL
    self.declarer = 0
    self.last_actor = 0
    self.bidder = 0
    self.flecks = 0
    self.passes = 0
    self.any_call = False
    self.bids: list[int] = []
    self.plays: list[int] = []
    self.play_seat: list[int] = []
    self.declared: list[bool] = []
    self.play: Optional[tuple] = None
    self.returns: Optional[list[float]] = None

  # -- queries --

  def is_chance(self) -> bool:
    return self.phase in (DEAL_CALLER, DEAL_REST)

  def is_terminal(self) -> bool:
    return self.phase == OVER

  def current_player(self) -> int:
    if self.phase == CHOOSE_TRUMP:
      return 0
    if self.phase == BIDDING:
      return self.bidder
    if self.phase == PLAY:
      return to_move(self.play)
    return -1

  def hands(self) -> tuple[int, ...]:
    """Each seat's remaining cards, as bitmasks."""
    if self.play is not None:
      return self.play[0]
    out = [0] * N_PLAYER
    for c, h in enumerate(self.holder):
      if h >= 0:
        out[h] |= 1 << c
    return tuple(out)

  def rules(self) -> Rules:
    return Rules(self.trump, self.contract, self.declarer)

  def deal_recipient(self) -> int:
    if self.phase == DEAL_CALLER:
      return 0
    k = self.n_dealt - CALLER_CARDS
    return 0 if k < 2 else 1 + (k - 2) // 5

  def bid_state(self) -> tuple:
    return (
        self.contract,
        self.declarer,
        self.last_actor,
        self.bidder,
        self.flecks,
        self.passes,
        self.any_call,
    )

  def legal_actions(self) -> list[int]:
    if self.is_chance():
      return [c for c in range(N_CARD) if self.holder[c] < 0]
    if self.phase == CHOOSE_TRUMP:
      return list(range(FIRST_TRUMP, FIRST_TRUMP + N_SUIT))
    if self.phase == BIDDING:
      return bid_legal(self.bid_state(), self.hands(), self.trump,
                       self.max_flecks)
    if self.phase == PLAY:
      return play_legal(self.rules(), self.play)
    return []

  # -- moves --

  def apply(self, action: int) -> None:
    if self.is_chance():
      assert self.holder[action] < 0, action
      self.holder[action] = self.deal_recipient()
      if self.phase == DEAL_CALLER:
        self.first_cards.append(action)
      self.n_dealt += 1
      if self.n_dealt == CALLER_CARDS:
        self.phase = CHOOSE_TRUMP
      elif self.n_dealt == N_CARD:
        self.phase = BIDDING
      return
    assert action in self.legal_actions(), (action, self.legal_actions())
    if self.phase == CHOOSE_TRUMP:
      self.trump = action - FIRST_TRUMP
      self.phase = DEAL_REST
      return
    if self.phase == BIDDING:
      self.bids.append(action - PASS)
      b, ended = bid_apply(self.bid_state(), action)
      (self.contract, self.declarer, self.last_actor, self.bidder,
       self.flecks, self.passes, self.any_call) = b
      if ended:
        if self.contract in (JODLER, HERMORT):
          self._finish(_contract_end(self.rules(), team_of(self.declarer)))
        else:
          self.phase = PLAY
          self.play = initial_play(self.hands(), self.rules())
      return
    seat = to_move(self.play)
    card = action
    if FIRST_DECLARE <= action < FIRST_TRUMP:
      idx = action - FIRST_DECLARE
      card = (idx // 2) * N_RANK + (KOENIG if idx % 2 == 0 else OBER)
    self.plays.append(card)
    self.play_seat.append(seat)
    self.declared.append(card != action)
    self.play, v = play_apply(self.rules(), self.play, action)
    if v is not None:
      self._finish(v)

  def _finish(self, value_team0: int) -> None:
    v = float(value_team0 * (1 << self.flecks))
    self.returns = [v if team_of(p) == 0 else -v for p in range(N_PLAYER)]
    self.phase = OVER

  # -- a seat's view, for a JAX rule bot --

  def info_state(self, seat: int) -> dict[str, np.ndarray]:
    """The seat's InfoState (envs.schnaps INFOSTATE), as numpy arrays."""
    nb = max_bids(self.max_flecks)
    bids = np.full(nb, -1, np.int8)
    bids[: len(self.bids)] = self.bids
    plays = np.full(N_CARD, -1, np.int8)
    plays[: len(self.plays)] = self.plays
    play_seat = np.full(N_CARD, -1, np.int8)
    play_seat[: len(self.play_seat)] = self.play_seat
    declared = np.zeros(N_CARD, bool)
    for c, d in zip(self.plays, self.declared):
      declared[c] = d
    dealt = np.array([h == seat for h in self.holder])
    played = np.zeros(N_CARD, bool)
    played[self.plays] = True
    first = np.zeros(N_CARD, bool)
    if seat == 0:
      first[self.first_cards] = True
    if self.play is not None:
      score, tricks, pending = self.play[3], self.play[4], self.play[5]
    else:
      score = tricks = pending = (0, 0)
    i8 = np.int8
    return {
        "seat": i8(seat),
        "phase": i8(self.phase),
        "trump": i8(self.trump),
        "contract": i8(self.contract),
        "declarer": i8(self.declarer),
        "last_actor": i8(self.last_actor),
        "flecks": i8(self.flecks),
        "passes": i8(self.passes),
        "any_call": np.bool_(self.any_call),
        "bids": bids,
        "n_bids": i8(len(self.bids)),
        "first_cards": first,
        "dealt": dealt,
        "hand": dealt & ~played,
        "plays": plays,
        "play_seat": play_seat,
        "declared": declared,
        "n_played": i8(len(self.plays)),
        "team_score": np.asarray(score, np.int32),
        "pending": np.asarray(pending, np.int32),
        "team_tricks": np.asarray(tricks, np.int8),
    }

  def legal_mask(self) -> np.ndarray:
    mask = np.zeros(39, bool)
    mask[self.legal_actions()] = True
    return mask


# -------------------------------- Bidding ----------------------------------


def _holds_suit(hand: int, s: int) -> bool:
  full = ((1 << N_RANK) - 1) << (s * N_RANK)
  return hand & full == full


def bid_legal(b: tuple, hands, trump: int, max_flecks: int) -> list[int]:
  """The legal bids (SchnapsState::BiddingActions)."""
  contract, _, last_actor, bidder, flecks, _, _ = b
  actions = [PASS]
  if flecks < max_flecks and team_of(last_actor) != team_of(bidder):
    actions.append(FLECK)
  hand = hands[bidder]
  for c in range(contract + 1, 6):
    if c == JODLER and not any(
        _holds_suit(hand, s) for s in range(N_SUIT) if s != trump
    ):
      continue
    if c == HERMORT and not _holds_suit(hand, trump):
      continue
    actions.append(FIRST_CALL + c - 1)
  return actions


def bid_apply(b: tuple, action: int) -> tuple[tuple, bool]:
  """Applies a bid: (the next bid state, whether the bidding ended)."""
  contract, declarer, last_actor, bidder, flecks, passes, any_call = b
  if action == PASS:
    passes += 1
  elif action == FLECK:
    flecks += 1
    last_actor = bidder
    passes = 0
    any_call = True
  else:
    contract = action - FIRST_CALL + 1
    declarer = bidder
    last_actor = bidder
    flecks = 0
    passes = 0
    any_call = True
  bidder = (bidder + 1) % N_PLAYER
  b = (contract, declarer, last_actor, bidder, flecks, passes, any_call)
  return b, passes >= (3 if any_call else 4)


# --------------------------------- Oracle ----------------------------------


class Oracle:
  """Perfect-information decisions for one deal (module doc).

  Make one per deal, after the cards are dealt; it caches its searches.
  """

  def __init__(self, hands: tuple[int, ...], trump: int, max_flecks: int = 2):
    self.hands = tuple(hands)
    self.trump = trump
    self.max_flecks = max_flecks
    self._play_solvers: dict[Rules, PlaySolver] = {}
    self._play_values: dict[Rules, int] = {}
    self._bid_values: dict[tuple, int] = {}

  def solver(self, rules: Rules) -> PlaySolver:
    if rules not in self._play_solvers:
      self._play_solvers[rules] = PlaySolver(rules)
    return self._play_solvers[rules]

  def contract_value(self, contract: int, declarer: int) -> int:
    """The value for team 0 of playing a contract, before flecks."""
    rules = Rules(self.trump, contract, declarer if contract else 0)
    if contract in (JODLER, HERMORT):
      return _contract_end(rules, team_of(declarer))
    if rules not in self._play_values:
      p = initial_play(self.hands, rules)
      self._play_values[rules] = self.solver(rules).value(p)
    return self._play_values[rules]

  def bid_value(self, b: tuple) -> int:
    """The bid state's value for team 0 under perfect play by all."""
    if b in self._bid_values:
      return self._bid_values[b]
    maximise = team_of(b[3]) == 0
    best = -INF if maximise else INF
    for a in bid_legal(b, self.hands, self.trump, self.max_flecks):
      v = self._bid_child(b, a)
      best = max(best, v) if maximise else min(best, v)
    self._bid_values[b] = best
    return best

  def _bid_child(self, b: tuple, a: int) -> int:
    q, ended = bid_apply(b, a)
    if not ended:
      return self.bid_value(q)
    contract, declarer, flecks = q[0], q[1], q[4]
    return self.contract_value(contract, declarer) * (1 << flecks)

  def action_values(self, model: Model) -> dict[int, int]:
    """Each legal action's exact value for team 0 (bidding or play)."""
    if model.phase == BIDDING:
      b = model.bid_state()
      return {
          a: self._bid_child(b, a)
          for a in bid_legal(b, self.hands, self.trump, self.max_flecks)
      }
    if model.phase == PLAY:
      scale = 1 << model.flecks
      vals = self.solver(model.rules()).action_values(model.play)
      return {a: v * scale for a, v in vals.items()}
    raise ValueError(f"no oracle decision in phase {model.phase}")

  def decide(self, model: Model) -> Optional[int]:
    """The oracle's action, or None at the trump choice (module doc).

    Among equally good actions it prefers a pass in the bidding, and in the
    play the cheapest card (a non-trump, low rank), declaring when that card
    allows it.
    """
    if model.phase == CHOOSE_TRUMP:
      return None
    vals = self.action_values(model)
    sign = 1 if team_of(model.current_player()) == 0 else -1
    best = max(sign * v for v in vals.values())
    tied = [a for a, v in vals.items() if sign * v == best]
    return min(tied, key=lambda a: tie_key(a, self.trump))


def action_values_with_hands(
    model: Model, hands: tuple[int, ...], max_flecks: int = 2
) -> dict[int, int]:
  """Each legal action's exact value for team 0 if `hands` were the hands.

  `hands` are the remaining cards of each seat (bitmasks), e.g. one sampled
  deal; the public state (bids, plays, scores) is the model's.
  """
  o = Oracle(hands, model.trump, max_flecks)
  if model.phase == PLAY:
    p = (tuple(hands),) + model.play[1:]
    scale = 1 << model.flecks
    vals = o.solver(model.rules()).action_values(p)
    return {a: v * scale for a, v in vals.items()}
  return o.action_values(model)


def tie_key(action: int, trump: int):
  """The order among equally good actions: pass, then the cheapest card."""
  if action == PASS:
    return (-1, 0)
  if action < FIRST_DECLARE:
    card, declare = action, 1
  elif action < FIRST_TRUMP:
    idx = action - FIRST_DECLARE
    card, declare = (idx // 2) * N_RANK + (KOENIG if idx % 2 == 0 else OBER), 0
  else:
    return (100 + action, 0)
  return (rank_of(card) * 2 + (suit_of(card) == trump), declare)
