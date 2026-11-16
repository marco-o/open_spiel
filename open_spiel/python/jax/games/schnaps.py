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

"""Schnaps (Bauernschnapsen) in pure JAX: OpenSpiel's ``schnaps``.

Four players in two partnerships (seats 0 and 2 against seats 1 and 3), a
20-card deck, five tricks. Encodings are OpenSpiel's (games/schnaps), so
actions, chance outcomes, players and returns compare with it without
translation:

- card = suit * 5 + rank, suit 0..3 = Herz, Caro, Pik, Kreuz, rank 0..4 =
  Unter, Ober, Koenig, Zehn, As, worth 2, 3, 4, 10 and 11 points;
- actions 0..19 are cards (dealt by chance, played by players), 20..27
  declare 20 or 40 and lead the Koenig (even) or Ober (odd) of suit
  (action - 20) // 2, 28..31 choose trump, 32 pass, 33 fleck, 34..38 call
  Schnapser, Lomt, Bauern, Jodler or Hermort.

Rules (OpenSpiel's schnaps.cc). Chance deals three cards to seat 0, the
Caller, who chooses trump; then two more to seat 0 and five each to seats 1,
2 and 3. Bidding starts at seat 0 and goes round: a player passes, flecks (at
most ``max_flecks`` times per contract, only against the other partnership's
last call or fleck; the Normal game counts as seat 0's) or calls a higher
contract (Jodler needs a whole non-trump suit, Hermort the whole trump
suit). A call resets the flecks. Bidding ends after three passes following
the last call or fleck, or four passes with none. (With ``normal_only``, the
second OpenSpiel parameter, pass is the only bid, so every deal is a Normal
game.) Jodler and Hermort win at once; otherwise seat 0 leads (the declarer in
Lomt). Players follow suit
and must beat the led suit's best card unless a trump has been played; a
player void in the led suit must trump, overtrumping when possible. Lomt is
played without trump. A leader holding the Koenig and Ober of a suit may
declare 20 (40 in trump) in Normal and Schnapser; the points count once the
partnership has taken a trick.

Scoring. Normal: the first team to 66 points (or the last trick's winner)
scores 3 if the losers took no trick, 2 if they have under 33 points, else 1.
Schnapser (6): the declarer's team wins each of the first three tricks and
reaches 66 by then (a trick won by the other side ends it). Lomt (9) and
Bauern (12): the declarer takes every trick. Jodler 18, Hermort 24. The
score doubles per fleck; returns are zero-sum, the winners' score for both
winners and its negative for the losers.

Information state. OpenSpiel's schnaps has no tensors, so
``information_state_tensor`` is this module's own perfect-recall view, a
function of OpenSpiel's information-state string (the tests check both
directions): the seat, the phase, the Caller's first three cards (seat 0's
view only), the cards dealt to the seat, its hand now, trump, each team's
points, pending declarations and tricks, every bid in order (``max_bids``
slots of seven: pass, fleck, the five calls), and every card played in order
(20 slots of the card and whether it was declared). ``observation_tensor``
is the same tensor.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_RANKS = 5
NUM_CARDS = 20
NUM_TRICKS = 5
CALLER_CARDS = 3
UNTER, OBER, KOENIG, ZEHN, AS = range(5)
RANK_POINTS = (2, 3, 4, 10, 11)

NORMAL, SCHNAPSER, LOMT, BAUERN, JODLER, HERMORT = range(6)
CONTRACT_POINTS = (3, 6, 9, 12, 18, 24)
CONTRACT_NAMES = ("Normal", "Schnapser", "Lomt", "Bauern", "Jodler", "Hermort")

FIRST_DECLARE_ACTION = 20
NUM_DECLARE_ACTIONS = 8
FIRST_TRUMP_ACTION = 28
PASS_ACTION = 32
FLECK_ACTION = 33
FIRST_CALL_ACTION = 34
NUM_DISTINCT_ACTIONS = 39
NUM_BID_KINDS = NUM_DISTINCT_ACTIONS - PASS_ACTION  # pass, fleck, 5 calls

DEFAULT_MAX_FLECKS = 2
MAX_MAX_FLECKS = 6
WINNING_POINTS = 66

(
    PHASE_DEAL_CALLER,
    PHASE_CHOOSE_TRUMP,
    PHASE_DEAL_REST,
    PHASE_BIDDING,
    PHASE_PLAY,
    PHASE_OVER,
) = range(6)
NUM_PHASES = 6

# The information-state tensor's fixed head.
PLAYER_OFFSET = 0
PHASE_OFFSET = PLAYER_OFFSET + NUM_PLAYERS  # 4
FIRST_CARDS_OFFSET = PHASE_OFFSET + NUM_PHASES  # 10
DEALT_OFFSET = FIRST_CARDS_OFFSET + NUM_CARDS  # 30
HAND_OFFSET = DEALT_OFFSET + NUM_CARDS  # 50
TRUMP_OFFSET = HAND_OFFSET + NUM_CARDS  # 70
SCORE_OFFSET = TRUMP_OFFSET + NUM_SUITS  # 74
BIDS_OFFSET = SCORE_OFFSET + 6  # 80
PLAY_SLOT_SIZE = NUM_CARDS + 1  # the card, then whether it was declared

PARAMS = ("max_flecks",)


def max_bids(max_flecks: int) -> int:
  """The most bids (passes included) one bidding round can take.

  At most five calls and `max_flecks` flecks per contract (six contracts),
  at most two passes between them, three before the first and three after
  the last; four passes when nothing is called. OpenSpiel's MaxGameLength
  uses the same bound.

  Args:
    max_flecks: The game parameter.

  Returns:
    The bound.
  """
  non_pass = 5 + 6 * max_flecks
  return 3 * non_pass + 4


def max_game_length(max_flecks: int) -> int:
  """OpenSpiel's MaxGameLength: deals, the trump choice, bids and plays.

  Args:
    max_flecks: The game parameter.

  Returns:
    The length.
  """
  return NUM_CARDS + 1 + max_bids(max_flecks) + NUM_CARDS


def info_state_size(max_flecks: int) -> int:
  """The length of the information-state tensor (module doc).

  Args:
    max_flecks: The game parameter.

  Returns:
    The length.
  """
  return (
      BIDS_OFFSET
      + max_bids(max_flecks) * NUM_BID_KINDS
      + NUM_CARDS * PLAY_SLOT_SIZE
  )


class GameState(NamedTuple):
  """A deal of schnaps. -1 marks an unset card, seat, suit or slot."""

  phase: jax.Array  # int8
  # int32: seat, CHANCE_PLAYER_ID, or TERMINAL_PLAYER_ID
  current_player: jax.Array
  is_terminal: jax.Array  # bool
  n_dealt: jax.Array  # int8
  holder: jax.Array  # int8[20]: the seat dealt each card
  first_cards: jax.Array  # bool[20]: the Caller's first three cards
  played: jax.Array  # bool[20]
  trump: jax.Array  # int8
  contract: jax.Array  # int8
  declarer: jax.Array  # int8, 0 for the Normal game
  last_actor: jax.Array  # int8: the last seat to call or fleck (0 at first)
  flecks: jax.Array  # int8
  passes: jax.Array  # int8: passes since the last call or fleck
  any_call: jax.Array  # bool: any call or fleck made
  bids: jax.Array  # int8[max_bids]: each bid as action - PASS_ACTION
  n_bids: jax.Array  # int32
  plays: jax.Array  # int8[20]: the cards in order of play
  play_seat: jax.Array  # int8[20]: who played each of them
  declared: jax.Array  # bool[20]: whether each was led with a declaration
  n_played: jax.Array  # int8
  team_score: jax.Array  # int32[2]: card points and valid declarations
  pending: jax.Array  # int32[2]: declarations awaiting the team's first trick
  team_tricks: jax.Array  # int8[2]
  points: jax.Array  # int32[4]: the returns, once terminal


def _i8(v) -> jax.Array:
  """`v` as int8.

  Args:
    v: A value or array.

  Returns:
    The int8 array.
  """
  return jnp.asarray(v, jnp.int8)


def _i32(v) -> jax.Array:
  """`v` as int32.

  Args:
    v: A value or array.

  Returns:
    The int32 array.
  """
  return jnp.asarray(v, jnp.int32)


def _cards() -> jax.Array:
  """The card ids, built at use (no array is made at import).

  Returns:
    int32[20], 0..19.
  """
  return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _rank_points() -> jax.Array:
  """The points of each rank, built at use.

  Returns:
    int32[5].
  """
  return jnp.asarray(RANK_POINTS, jnp.int32)


def _contract_points() -> jax.Array:
  """The base score of each contract, built at use.

  Returns:
    int32[6].
  """
  return jnp.asarray(CONTRACT_POINTS, jnp.int32)


def _where_tree(cond, x, y):
  """`x` where `cond`, else `y`, leaf by leaf.

  Args:
    cond: A bool scalar.
    x: A pytree.
    y: A pytree of the same structure.

  Returns:
    The selected pytree.
  """
  return jax.tree.map(lambda a, b: jnp.where(cond, a, b), x, y)


def card_suit(card):
  """The suit of a card (card // 5).

  Args:
    card: The card id(s).

  Returns:
    The suit(s).
  """
  return card // NUM_RANKS


def card_rank(card):
  """The rank of a card (card % 5), 0..4 = Unter..As.

  Args:
    card: The card id(s).

  Returns:
    The rank(s).
  """
  return card % NUM_RANKS


def declared_card(action):
  """The card a declaration leads: its suit's Koenig (even) or Ober (odd).

  Args:
    action: The declaration action(s), 20..27.

  Returns:
    The card id(s).
  """
  index = action - FIRST_DECLARE_ACTION
  return (index // 2) * NUM_RANKS + jnp.where(index % 2 == 0, KOENIG, OBER)


def info_state_tensor(
    *,
    seat,
    phase,
    first_cards,
    dealt,
    hand,
    trump,
    team_score,
    pending,
    team_tricks,
    bids,
    plays,
    declared,
) -> jax.Array:
  """The information-state tensor of one seat (module doc).

  Args:
    seat: The observing seat.
    phase: The phase.
    first_cards: bool[20], the Caller's first three cards; read only in seat
      0's view.
    dealt: bool[20], the cards dealt to the seat.
    hand: bool[20], the cards it holds now.
    trump: The trump suit, -1 before it is chosen.
    team_score: int[2], each team's points.
    pending: int[2], each team's declarations awaiting its first trick.
    team_tricks: int[2], each team's tricks.
    bids: int[max_bids], each bid as action - PASS_ACTION (-1 pad); its
      length sets the bid slots.
    plays: int[20], the cards in order of play (-1 pad).
    declared: bool[20], whether each was led with a declaration.

  Returns:
    f32[info_state_size(max_flecks)].
  """
  f32 = jnp.float32
  seat = _i32(seat)
  plays = jnp.concatenate(
      [
          jax.nn.one_hot(_i32(plays), NUM_CARDS, dtype=f32),
          jnp.asarray(declared).astype(f32)[:, None],
      ],
      axis=1,
  )
  return jnp.concatenate([
      jax.nn.one_hot(seat, NUM_PLAYERS, dtype=f32),
      jax.nn.one_hot(_i32(phase), NUM_PHASES, dtype=f32),
      (jnp.asarray(first_cards) & (seat == 0)).astype(f32),
      jnp.asarray(dealt).astype(f32),
      jnp.asarray(hand).astype(f32),
      jax.nn.one_hot(_i32(trump), NUM_SUITS, dtype=f32),
      _i32(team_score).astype(f32) / WINNING_POINTS,
      _i32(pending).astype(f32) / WINNING_POINTS,
      _i32(team_tricks).astype(f32) / NUM_TRICKS,
      jax.nn.one_hot(_i32(bids), NUM_BID_KINDS, dtype=f32).reshape(-1),
      plays.reshape(-1),
  ])


class Schnaps(game_lib.Game, interop.PyspielInterop):
  """OpenSpiel schnaps (module doc); the parameter is OpenSpiel's."""

  def __init__(
      self, max_flecks: int = DEFAULT_MAX_FLECKS, normal_only: bool = False
  ):
    if not isinstance(normal_only, bool):
      raise TypeError(
          f"schnaps parameter normal_only must be a bool, got {normal_only!r}"
      )
    if isinstance(max_flecks, bool) or not isinstance(max_flecks, int):
      raise TypeError(
          f"schnaps parameter max_flecks must be an int, got {max_flecks!r}"
      )
    if not 0 <= max_flecks <= MAX_MAX_FLECKS:
      raise ValueError(
          f"schnaps parameter max_flecks must be in 0..{MAX_MAX_FLECKS}, "
          f"got {max_flecks}"
      )
    self.params = {"max_flecks": max_flecks}
    if normal_only:  # named only when set, as OpenSpiel game strings do
      self.params["normal_only"] = True
    self.max_flecks = max_flecks
    self.normal_only = bool(normal_only)
    self.max_bids = max_bids(max_flecks)

  @property
  def num_actions(self) -> int:
    """The 20 cards, 8 declarations, 4 trump suits, pass, fleck, 5 calls."""
    return NUM_DISTINCT_ACTIONS

  @property
  def num_players(self) -> int:
    """Four seats."""
    return NUM_PLAYERS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The information-state tensor shape."""
    return (info_state_size(self.max_flecks),)

  @property
  def max_chance_outcomes(self) -> int:
    """The 20 cards."""
    return NUM_CARDS

  @property
  def max_game_length(self) -> int:
    """OpenSpiel's MaxGameLength (`max_game_length`)."""
    return max_game_length(self.max_flecks)

  def new_initial_state(self) -> GameState:
    """The state before the first card is dealt.

    Returns:
      The initial GameState (a chance node).
    """
    return GameState(
        phase=_i8(PHASE_DEAL_CALLER),
        current_player=_i32(game_lib.CHANCE_PLAYER_ID),
        is_terminal=jnp.asarray(False),
        n_dealt=_i8(0),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        first_cards=jnp.zeros((NUM_CARDS,), jnp.bool_),
        played=jnp.zeros((NUM_CARDS,), jnp.bool_),
        trump=_i8(-1),
        contract=_i8(NORMAL),
        declarer=_i8(0),
        last_actor=_i8(0),
        flecks=_i8(0),
        passes=_i8(0),
        any_call=jnp.asarray(False),
        bids=jnp.full((self.max_bids,), -1, jnp.int8),
        n_bids=_i32(0),
        plays=jnp.full((NUM_CARDS,), -1, jnp.int8),
        play_seat=jnp.full((NUM_CARDS,), -1, jnp.int8),
        declared=jnp.zeros((NUM_CARDS,), jnp.bool_),
        n_played=_i8(0),
        team_score=jnp.zeros((2,), jnp.int32),
        pending=jnp.zeros((2,), jnp.int32),
        team_tricks=jnp.zeros((2,), jnp.int8),
        points=jnp.zeros((NUM_PLAYERS,), jnp.int32),
    )

  # -- transitions --------------------------------------------------------

  def _deal_step(self, s: GameState, card) -> GameState:
    """Deals `card` to the next seat (OpenSpiel's DealRecipient).

    Args:
      s: The state.
      card: The card.

    Returns:
      The state; after three cards the Caller chooses trump, after twenty
      bidding starts.
    """
    n = _i32(s.n_dealt)
    to_caller = s.phase == PHASE_DEAL_CALLER
    k = n - CALLER_CARDS
    recipient = jnp.where(
        to_caller | (k < 2), 0, 1 + (k - 2) // NUM_TRICKS
    )
    n1 = n + 1
    to_trump = to_caller & (n1 == CALLER_CARDS)
    to_bid = ~to_caller & (n1 == NUM_CARDS)
    phase = jnp.where(
        to_trump, PHASE_CHOOSE_TRUMP, jnp.where(to_bid, PHASE_BIDDING, s.phase)
    )
    return s._replace(
        holder=s.holder.at[card].set(_i8(recipient)),
        first_cards=jnp.where(
            to_caller, s.first_cards.at[card].set(True), s.first_cards
        ),
        n_dealt=_i8(n1),
        phase=_i8(phase),
        current_player=_i32(
            jnp.where(to_trump | to_bid, 0, game_lib.CHANCE_PLAYER_ID)
        ),
    )

  def _trump_step(self, s: GameState, a) -> GameState:
    """The Caller chooses trump; the rest of the cards are dealt.

    Args:
      s: The state.
      a: The trump action.

    Returns:
      The state, dealing.
    """
    return s._replace(
        trump=_i8(jnp.clip(a - FIRST_TRUMP_ACTION, 0, NUM_SUITS - 1)),
        phase=_i8(PHASE_DEAL_REST),
        current_player=_i32(game_lib.CHANCE_PLAYER_ID),
    )

  def _finish(self, s: GameState, team, base) -> GameState:
    """OpenSpiel FinishGame: `team` wins `base` doubled per fleck.

    Args:
      s: The state.
      team: The winning partnership, 0 or 1.
      base: The contract's base score.

    Returns:
      The terminal state.
    """
    value = _i32(base) * jnp.left_shift(1, _i32(s.flecks))
    winners = jnp.arange(NUM_PLAYERS) % 2 == team
    return s._replace(
        points=_i32(jnp.where(winners, value, -value)),
        phase=_i8(PHASE_OVER),
        current_player=_i32(game_lib.TERMINAL_PLAYER_ID),
    )

  def _normal_base(self, s: GameState, winner) -> jax.Array:
    """OpenSpiel FinishNormal's score: 3, 2 or 1 by what the losers took.

    Args:
      s: The state.
      winner: The winning partnership.

    Returns:
      The base score.
    """
    loser = 1 - winner
    return jnp.where(
        s.team_tricks[loser] == 0,
        3,
        jnp.where(s.team_score[loser] < WINNING_POINTS // 2, 2, 1),
    )

  def _game_end(
      self, s: GameState, after_trick: bool, last_winner, tricks_played
  ) -> tuple[jax.Array, GameState]:
    """OpenSpiel CheckGameEnd: whether the deal is over, and its end.

    Args:
      s: The state, scores updated.
      after_trick: Whether a trick was just taken (else a declaration).
      last_winner: The seat that took the last trick (after a trick).
      tricks_played: The tricks taken so far (after a trick).

    Returns:
      (ended, the finished state); the state is meaningful only if ended.
    """
    contract = _i32(s.contract)
    side = _i32(s.declarer) % 2
    score = s.team_score
    reached = score >= WINNING_POINTS
    # Normal: the first team to 66, else the last trick's winner.
    normal_w = jnp.where(reached[0], 0, 1)
    normal_end = reached.any()
    # Schnapser: the declarer's team wins each of the first three tricks
    # and reaches 66 by then.
    schnapser_end = reached[side]
    schnapser_w = side
    if after_trick:
      last_team = _i32(last_winner) % 2
      normal_w = jnp.where(normal_end, normal_w, last_team)
      normal_end = normal_end | (tricks_played == NUM_TRICKS)
      schnapser_w = jnp.where(schnapser_end, side, 1 - side)
      # A trick won by the other side ends it, as does the third trick.
      schnapser_end = (
          schnapser_end
          | (tricks_played == 3)
          | (_i32(last_winner) % 2 != side)
      )
      # Lomt and Bauern: the declarer takes every trick.
      lost = _i32(last_winner) != _i32(s.declarer)
      all_end = lost | (tricks_played == NUM_TRICKS)
      all_w = jnp.where(lost, 1 - side, side)
    else:
      all_end = jnp.asarray(False)
      all_w = side
    is_normal = contract == NORMAL
    is_schnapser = contract == SCHNAPSER
    is_all = (contract == LOMT) | (contract == BAUERN)
    ended = (
        (is_normal & normal_end)
        | (is_schnapser & schnapser_end)
        | (is_all & all_end)
    )
    winner = jnp.where(
        is_normal, normal_w, jnp.where(is_schnapser, schnapser_w, all_w)
    )
    base = jnp.where(
        is_normal,
        self._normal_base(s, winner),
        _contract_points()[jnp.clip(contract, 0, HERMORT)],
    )
    return ended, self._finish(s, winner, base)

  def _bid_step(self, s: GameState, a) -> GameState:
    """The current player passes, flecks or calls (OpenSpiel ApplyBid).

    Args:
      s: The state.
      a: The bid.

    Returns:
      The state; still bidding, in play, or over (Jodler, Hermort).
    """
    p = s.current_player
    is_pass = a == PASS_ACTION
    is_fleck = a == FLECK_ACTION
    is_call = (a >= FIRST_CALL_ACTION) & (a < NUM_DISTINCT_ACTIONS)
    passes = jnp.where(is_pass, _i32(s.passes) + 1, 0)
    contract = jnp.where(is_call, a - FIRST_CALL_ACTION + 1, s.contract)
    declarer = jnp.where(is_call, p, s.declarer)
    any_call = s.any_call | is_fleck | is_call
    b = s._replace(
        passes=_i8(passes),
        flecks=_i8(
            jnp.where(
                is_fleck, _i32(s.flecks) + 1, jnp.where(is_call, 0, s.flecks)
            )
        ),
        contract=_i8(contract),
        declarer=_i8(declarer),
        last_actor=_i8(jnp.where(is_fleck | is_call, p, s.last_actor)),
        any_call=any_call,
        bids=s.bids.at[s.n_bids].set(_i8(a - PASS_ACTION), mode="drop"),
        n_bids=s.n_bids + 1,
        current_player=_i32((p + 1) % NUM_PLAYERS),
    )
    end = passes >= jnp.where(any_call, 3, 4)
    contract = _i32(contract)
    resolved = (contract == JODLER) | (contract == HERMORT)
    finished = self._finish(
        b, _i32(declarer) % 2, _contract_points()[contract]
    )
    leader = jnp.where(contract == LOMT, declarer, 0)
    play = b._replace(phase=_i8(PHASE_PLAY), current_player=_i32(leader))
    return _where_tree(end, _where_tree(resolved, finished, play), b)

  def _play_step(self, s: GameState, a) -> GameState:
    """The current player leads or plays a card (OpenSpiel ApplyPlay).

    A declaration scores (or pends) its 20 or 40 first; if that ends the
    deal, the card is not played.

    Args:
      s: The state.
      a: A card or a declaration.

    Returns:
      The state.
    """
    p = s.current_player
    is_decl = (a >= FIRST_DECLARE_ACTION) & (
        a < FIRST_DECLARE_ACTION + NUM_DECLARE_ACTIONS
    )
    card = jnp.where(is_decl, declared_card(a), jnp.clip(a, 0, NUM_CARDS - 1))
    team = p % 2
    decl_points = _i32(jnp.where(card_suit(card) == s.trump, 40, 20))
    counts = s.team_tricks[team] > 0
    s1 = s._replace(
        team_score=jnp.where(
            is_decl & counts, s.team_score.at[team].add(decl_points),
            s.team_score,
        ),
        pending=jnp.where(
            is_decl & ~counts, s.pending.at[team].add(decl_points), s.pending
        ),
    )
    decl_ended, decl_finished = self._game_end(s1, False, -1, 0)

    t = _i32(s.n_played)
    plays = s.plays.at[t].set(_i8(card))
    play_seat = s.play_seat.at[t].set(_i8(p))
    s2 = s1._replace(
        played=s.played.at[card].set(True),
        plays=plays,
        play_seat=play_seat,
        declared=s.declared.at[t].set(is_decl),
        n_played=_i8(t + 1),
        current_player=_i32((p + 1) % NUM_PLAYERS),
    )
    trick_done = (t + 1) % NUM_PLAYERS == 0
    k = t // NUM_PLAYERS
    cards = jax.lax.dynamic_slice(_i32(plays), (k * NUM_PLAYERS,), (4,))
    seats = jax.lax.dynamic_slice(_i32(play_seat), (k * NUM_PLAYERS,), (4,))
    suits = card_suit(cards)
    ranks = card_rank(cards)
    trump_on = s.contract != LOMT
    key = jnp.where(
        trump_on & (suits == s.trump),
        100 + ranks,
        jnp.where(suits == suits[0], ranks, -1),
    )
    winner = seats[jnp.argmax(key)]
    wteam = winner % 2
    gained = _i32(jnp.sum(_rank_points()[ranks]) + s1.pending[wteam])
    taken = s2._replace(
        team_score=s1.team_score.at[wteam].add(gained),
        pending=s1.pending.at[wteam].set(0),
        team_tricks=s.team_tricks.at[wteam].add(_i8(1)),
        current_player=_i32(winner),
    )
    trick_ended, trick_finished = self._game_end(taken, True, winner, k + 1)
    after_trick = _where_tree(trick_ended, trick_finished, taken)
    played = _where_tree(trick_done, after_trick, s2)
    return _where_tree(is_decl & decl_ended, decl_finished, played)

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Applies a chance outcome or the current player's action.

    A terminal state is returned unchanged. The action is not validated
    (OpenSpiel's engine rejects an illegal one; use `legal_actions_mask`).

    Args:
      state: The current state.
      action: The chance outcome or action.

    Returns:
      The next state.
    """
    a = _i32(action)
    card = jnp.clip(a, 0, NUM_CARDS - 1)
    candidates = (
        self._deal_step(state, card),
        self._trump_step(state, a),
        self._deal_step(state, card),
        self._bid_step(state, a),
        self._play_step(state, a),
        state,
    )

    def pick(*leaves):
      """The leaf of the phase's candidate state.

      Args:
        *leaves: The leaf of each candidate, in phase order, then the unchanged
          state's.

      Returns:
        The chosen leaf.
      """
      out = leaves[-1]
      for ph in range(PHASE_PLAY, PHASE_DEAL_CALLER - 1, -1):
        out = jnp.where(state.phase == ph, leaves[ph], out)
      return out

    nxt = jax.tree.map(pick, *candidates)
    return nxt._replace(is_terminal=nxt.phase == PHASE_OVER)

  # -- queries ------------------------------------------------------------

  def _bid_mask(self, s: GameState) -> jax.Array:
    """OpenSpiel BiddingActions.

    Args:
      s: The state, bidding.

    Returns:
      bool[39].
    """
    p = s.current_player
    acts = jnp.arange(NUM_DISTINCT_ACTIONS)
    if self.normal_only:  # the Normal-only variant: pass is the only bid
      return acts == PASS_ACTION
    can_fleck = (_i32(s.flecks) < self.max_flecks) & (
        _i32(s.last_actor) % 2 != p % 2
    )
    whole = (s.holder == p).reshape(NUM_SUITS, NUM_RANKS).all(axis=1)
    trump = _i32(s.trump)
    suits = jnp.arange(NUM_SUITS)
    jodler = jnp.any(whole & (suits != trump))
    hermort = jnp.any(whole & (suits == trump))
    called = acts - FIRST_CALL_ACTION + 1
    call = (
        (acts >= FIRST_CALL_ACTION)
        & (called > s.contract)
        & ((called != JODLER) | jodler)
        & ((called != HERMORT) | hermort)
    )
    return (acts == PASS_ACTION) | ((acts == FLECK_ACTION) & can_fleck) | call

  def _play_mask(self, s: GameState) -> jax.Array:
    """OpenSpiel PlayActions: lead anything, or follow, beat and trump.

    Args:
      s: The state, at a play.

    Returns:
      bool[39].
    """
    p = s.current_player
    hand = (s.holder == p) & ~s.played
    t = _i32(s.n_played)
    in_trick = t % NUM_PLAYERS
    contract = _i32(s.contract)
    trump = _i32(s.trump)
    trump_on = contract != LOMT
    # Leading: any card, and the declarations of the pairs held.
    by_suit = hand.reshape(NUM_SUITS, NUM_RANKS)
    pairs = by_suit[:, KOENIG] & by_suit[:, OBER]
    pairs &= (contract == NORMAL) | (contract == SCHNAPSER)
    lead = jnp.concatenate([
        hand,
        jnp.repeat(pairs, 2),
        jnp.zeros((NUM_DISTINCT_ACTIONS - FIRST_TRUMP_ACTION,), jnp.bool_),
    ])
    # Following.
    cards = jax.lax.dynamic_slice(_i32(s.plays), (t - in_trick,), (4,))
    valid = jnp.arange(NUM_PLAYERS) < in_trick
    suits = card_suit(cards)
    ranks = card_rank(cards)
    led = suits[0]
    is_trump = valid & trump_on & (suits == trump)
    best_trump = jnp.max(jnp.where(is_trump, ranks, -1))
    best_led = jnp.max(jnp.where(valid & (suits == led), ranks, -1))
    already_trumped = trump_on & is_trump.any() & (led != trump)
    all_suits = card_suit(_cards())
    all_ranks = card_rank(_cards())
    follow = hand & (all_suits == led)
    higher = follow & (all_ranks > best_led)
    follow = jnp.where(~already_trumped & higher.any(), higher, follow)
    trumps = hand & trump_on & (all_suits == trump)
    over = trumps & (all_ranks > best_trump)
    trumps = jnp.where(over.any(), over, trumps)
    chosen = jnp.where(
        follow.any(), follow, jnp.where(trumps.any(), trumps, hand)
    )
    following = jnp.concatenate([
        chosen,
        jnp.zeros((NUM_DISTINCT_ACTIONS - NUM_CARDS,), jnp.bool_),
    ])
    return jnp.where(in_trick == 0, lead, following)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """The current player's legal actions; the undealt cards at chance nodes.

    Args:
      state: The current state.

    Returns:
      bool[39].
    """
    acts = jnp.arange(NUM_DISTINCT_ACTIONS)
    pad = jnp.zeros((NUM_DISTINCT_ACTIONS - NUM_CARDS,), jnp.bool_)
    undealt = jnp.concatenate([state.holder < 0, pad])
    ph = state.phase
    return jnp.select(
        [
            (ph == PHASE_DEAL_CALLER) | (ph == PHASE_DEAL_REST),
            ph == PHASE_CHOOSE_TRUMP,
            ph == PHASE_BIDDING,
            ph == PHASE_PLAY,
        ],
        [
            undealt,
            (acts >= FIRST_TRUMP_ACTION) & (acts < PASS_ACTION),
            self._bid_mask(state),
            self._play_mask(state),
        ],
        jnp.zeros((NUM_DISTINCT_ACTIONS,), jnp.bool_),
    )

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """The chance distribution: uniform over the undealt cards.

    Args:
      state: The current state.

    Returns:
      f32[20].
    """
    undealt = (state.holder < 0).astype(jnp.float32)
    return undealt / jnp.maximum(undealt.sum(), 1.0)

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Whether the deal is over.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Whether a card is being dealt.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return (state.phase == PHASE_DEAL_CALLER) | (
        state.phase == PHASE_DEAL_REST
    )

  def returns(self, state: GameState) -> jax.Array:
    """The winners' score (negated for the losers) at the end; zeros before.

    Args:
      state: The current state.

    Returns:
      f32[4].
    """
    return jnp.where(state.is_terminal, state.points, 0).astype(jnp.float32)

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """The perfect-recall information-state tensor (module doc).

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[info_state_size(max_flecks)].
    """
    dealt = state.holder == observing_player
    return info_state_tensor(
        seat=observing_player,
        phase=state.phase,
        first_cards=state.first_cards,
        dealt=dealt,
        hand=dealt & ~state.played,
        trump=state.trump,
        team_score=state.team_score,
        pending=state.pending,
        team_tricks=state.team_tricks,
        bids=state.bids,
        plays=state.plays,
        declared=state.declared,
    )

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """The information-state tensor.

    OpenSpiel schnaps has no observation tensor.

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[info_state_size(max_flecks)].
    """
    return self.information_state_tensor(state, observing_player)

  def state_from_os(self, os_state: Any) -> GameState:
    """The JAX state of a pyspiel schnaps state, by replaying its history.

    The two games share action ids, so the history replays unchanged.

    Args:
      os_state: The pyspiel state.

    Returns:
      The GameState.
    """
    state = self.new_initial_state()
    for action in os_state.history():
      state = self.apply_action(state, jnp.int32(action))
    return state

  def information_state_from_os(self, os_state: Any, player: int) -> jax.Array:
    """This module's information-state tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[info_state_size(max_flecks)].
    """
    return self.information_state_tensor(
        self.state_from_os(os_state), jnp.int32(player)
    )

  def observation_from_os(self, os_state: Any, player: int) -> jax.Array:
    """The information-state tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[info_state_size(max_flecks)].
    """
    return self.information_state_from_os(os_state, player)
