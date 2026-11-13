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

"""A fair schnaps search bot: sample the hidden hands, solve each exactly.

Determinized search ("perfect-information Monte Carlo"). At each decision
the bot deals the cards it cannot see at random, `n_samples` times, in ways
consistent with what its seat knows, solves each deal with the oracle's
exact solver (envs.schnaps_oracle), and plays the action with the best mean
value over the deals. Unlike the oracle it never looks at another hand.

What a sampled deal respects (from the public plays and the seat's hand):
  * hand sizes: each seat holds 5 cards minus those it played;
  * a declared pair: a seat that declared with the Koenig (Ober) of a suit
    still holds the Ober (Koenig) if it has not played it;
  * voids: a seat that did not follow the led suit has none of it; one that
    also did not trump (in a contract with trump) has no trump;
  * must-beat: a seat that followed suit without beating the best card of
    the led suit (when no trump had been played on it) has no higher card of
    that suit; one that trumped without overtrumping has no higher trump.
What it ignores: what the bidding says about the hands (every consistent
deal is equally likely), and the Caller's three first cards, which only the
Caller sees (its own hand covers them).

Known weaknesses of the method: it assumes it will know every hand after
this move (so it values information it will not have), and in the bidding
it judges a contract as if played with open cards. The bidding is therefore
optional (`phases`): by default the bot only searches the card play and
leaves the trump choice and the bidding to its caller.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from open_spiel.python.scale.envs import schnaps_oracle as oracle

N_PLAYER = oracle.N_PLAYER
N_RANK = oracle.N_RANK


def _suit_cards(s: int) -> set[int]:
  return {s * N_RANK + r for r in range(N_RANK)}


def constraints(model: oracle.Model, seat: int):
  """What `seat` knows of the hidden cards.

  Returns:
    (known, allowed, need): known {card: holder} for hidden cards whose
    holder is certain, allowed {card: set of seats} for the other hidden
    cards, need [cards still to place per seat] (0 for `seat`).
  """
  hands = model.hands()
  mine = set(oracle.cards_of(hands[seat]))
  played = set(model.plays)
  hidden = set(range(oracle.N_CARD)) - mine - played
  others = [p for p in range(N_PLAYER) if p != seat]
  excluded = {p: set() for p in others}
  known: dict[int, int] = {}
  trump = model.trump
  active = model.contract != oracle.LOMT

  # Walk the plays trick by trick.
  for t in range(0, len(model.plays), N_PLAYER):
    cards = model.plays[t : t + N_PLAYER]
    seats = model.play_seat[t : t + N_PLAYER]
    led = oracle.suit_of(cards[0])
    best_led, best_trump, trumped = oracle.rank_of(cards[0]), -1, False
    if active and led == trump:
      trumped, best_trump = True, oracle.rank_of(cards[0])
    for i in range(1, len(cards)):
      c, p = cards[i], seats[i]
      s, r = oracle.suit_of(c), oracle.rank_of(c)
      if p != seat:
        ex = excluded[p]
        if s != led:
          ex |= _suit_cards(led)
          if active and s != trump:
            ex |= _suit_cards(trump)
          elif active and s == trump and r < best_trump:
            ex |= {trump * N_RANK + q for q in range(best_trump + 1, N_RANK)}
        elif not (active and trumped and led != trump) and r < best_led:
          ex |= {led * N_RANK + q for q in range(best_led + 1, N_RANK)}
      if s == led:
        best_led = max(best_led, r)
      if active and s == trump:
        trumped = True
        best_trump = max(best_trump, r)

  # Declared pairs: the other card of the pair stays with the declarer.
  for c, p, d in zip(model.plays, model.play_seat, model.declared):
    if d and p != seat:
      s, r = oracle.suit_of(c), oracle.rank_of(c)
      other = s * N_RANK + (oracle.OBER if r == oracle.KOENIG else oracle.KOENIG)
      if other in hidden:
        known[other] = p

  need = [0] * N_PLAYER
  n_played = [0] * N_PLAYER
  for p in model.play_seat:
    n_played[p] += 1
  for p in others:
    need[p] = oracle.HAND_SIZE - n_played[p]
  for c, p in known.items():
    need[p] -= 1
  allowed = {}
  for c in hidden - set(known):
    allowed[c] = {p for p in others if c not in excluded[p]}
  return known, allowed, need


def sample_hands(
    model: oracle.Model, seat: int, rng: random.Random, tries: int = 200
) -> Optional[tuple[int, ...]]:
  """One deal of the hidden cards consistent with `seat`'s view, or None.

  The most constrained cards are placed first, each with a seat drawn in
  proportion to its free places. The draw is not exactly uniform over the
  consistent deals, but close for the few constraints a deal carries.
  """
  known, allowed, need0 = constraints(model, seat)
  hands = list(model.hands())
  base = [0] * N_PLAYER
  base[seat] = hands[seat]
  for c, p in known.items():
    base[p] |= 1 << c
  cards = list(allowed)
  for _ in range(tries):
    rng.shuffle(cards)
    cards.sort(key=lambda c: len(allowed[c]))
    need = list(need0)
    out = list(base)
    ok = True
    for c in cards:
      seats = [p for p in allowed[c] if need[p] > 0]
      if not seats:
        ok = False
        break
      p = rng.choices(seats, weights=[need[q] for q in seats])[0]
      need[p] -= 1
      out[p] |= 1 << c
    if ok and not any(need):
      return tuple(out)
  return None


class SearchBot:
  """Determinized search over sampled deals (module doc)."""

  def __init__(
      self,
      n_samples: int = 20,
      seed: int = 0,
      phases: Sequence[int] = (oracle.PLAY,),
      max_flecks: int = 2,
  ):
    self.n_samples = n_samples
    self.rng = random.Random(seed)
    self.phases = tuple(phases)
    self.max_flecks = max_flecks
    self.failed_samples = 0

  def action_values(self, model: oracle.Model) -> dict[int, float]:
    """Each legal action's mean value for team 0 over the sampled deals."""
    seat = model.current_player()
    legal = model.legal_actions()
    total = {a: 0.0 for a in legal}
    n = 0
    for _ in range(self.n_samples):
      hands = sample_hands(model, seat, self.rng)
      if hands is None:
        self.failed_samples += 1
        continue
      vals = oracle.action_values_with_hands(model, hands, self.max_flecks)
      for a in legal:
        total[a] += vals[a]
      n += 1
    if n == 0:  # cannot happen with a consistent view; keep playing
      return {a: 0.0 for a in legal}
    return {a: v / n for a, v in total.items()}

  def decide(self, model: oracle.Model) -> Optional[int]:
    """The bot's action, or None where it does not search (`phases`)."""
    if model.phase not in self.phases:
      return None
    legal = model.legal_actions()
    if len(legal) == 1:
      return legal[0]
    vals = self.action_values(model)
    sign = 1 if oracle.team_of(model.current_player()) == 0 else -1
    best = max(sign * v for v in vals.values())
    tied = [a for a, v in vals.items() if sign * v >= best - 1e-9]
    return min(tied, key=lambda a: oracle.tie_key(a, model.trump))
