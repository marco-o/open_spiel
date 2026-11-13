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

"""Schnaps with a forced contract: card-play training for one called game.

Every deal is a Schnapser, a Lomt or a Bauern (one per env) whose declarer
holds a plausible hand. The deal is all chance, then play is schnaps's:

  1. chance picks the declarer's seat (uniform) and the trump (uniform; a
     Lomt shows one but plays without it);
  2. chance deals the declarer's five cards one at a time from the hand
     distribution W[seat, trump, hand] (below);
  3. chance deals the other fifteen cards uniformly, five to each other
     seat in seat order;
  4. the scenario is replayed into OpenSpiel's schnaps (the JAX engine
     jax.games.schnaps at max_flecks 2), one scripted chance node per base
     move: seat 0's first three cards (its trumps first, then in deal
     order), the trump choice, the rest of the deal in the engine's order,
     then the bidding: the seats before the declarer pass, the declarer calls
     the contract, the next three pass.

The state after step 4 is a genuine schnaps state at the first play, the one
a real deal reaches with those bids, so the InfoState is schnaps's and
schnaps's encoders and rule bots (Elena) read it unchanged. The only
decisions are card plays and declarations: no trump choice, no bids, no
flecks.

HAND DISTRIBUTION. A hand's class is its suit masks (bit r = rank r held,
ranks Unter..As) sorted canonically: for Schnapser and Bauern the trump
suit's mask first, then the other three descending (3,077 classes); for Lomt,
played without trump, all four descending (901 classes). The hands file
(``schnaps_forced_hands.json``, written by
examples/schnaps_forced_hands.py from the oracle's full-information solves)
gives each class a weight per declarer seat; every hand of a class carries
its class's weight, so a class is dealt in proportion to how often it occurs
times its weight. P(next declarer card = c) is the weight of the hands that
hold the cards dealt so far and c, over the weight of those that hold the
cards dealt so far.

Known simplification: when the declarer is not seat 0, the trump is drawn
independently of seat 0's hand (in a real deal the Caller usually names a
suit it holds).

RULES IDENTITY. ``declared_rules_sha``: the sha256 over canonical JSON of
schnaps's rules identity at max_flecks 2, the contract, and the hands file's
sha256.

Not served (envs.registry): a forced-contract network has never bid.
"""

import functools
import hashlib
import itertools
import json
import os
from typing import Any, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import schnaps
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.envs import schnaps as schnaps_env

NUM_CARDS = schnaps.NUM_CARDS
NUM_PLAYERS = schnaps.NUM_PLAYERS
NUM_SUITS = 4
NUM_RANKS = 5
HAND_SIZE = 5
MAX_FLECKS = schnaps.DEFAULT_MAX_FLECKS

#: The forced contracts, by env suffix.
CONTRACTS = {
    "schnapser": schnaps.SCHNAPSER,
    "lomt": schnaps.LOMT,
    "bauern": schnaps.BAUERN,
}

HANDS_FILE = os.path.join(os.path.dirname(__file__), "schnaps_forced_hands.json")
HANDS_SCHEMA = "schnaps-forced-hands/1"

# The stages of a forced deal.
STAGE_SEAT, STAGE_TRUMP, STAGE_DECLARER, STAGE_OTHERS, STAGE_REPLAY, STAGE_PLAY = (
    range(6)
)
#: The longest replay: 20 deals, the trump choice, 3 passes, the call, 3
#: passes.
MAX_REPLAY = NUM_CARDS + 1 + (NUM_PLAYERS - 1) + 1 + 3


# -- hands and classes (numpy) -------------------------------------------------


def _all_hands() -> np.ndarray:
  """Every five-card hand, as bool[15504, 20], in itertools order."""
  idx = np.array(list(itertools.combinations(range(NUM_CARDS), HAND_SIZE)))
  bits = np.zeros((len(idx), NUM_CARDS), np.bool_)
  bits[np.arange(len(idx))[:, None], idx] = True
  return bits


HAND_BITS = _all_hands()
NUM_HANDS = HAND_BITS.shape[0]
assert NUM_HANDS == 15504


def suit_masks(bits: np.ndarray) -> np.ndarray:
  """Each suit's rank mask (bit r = rank r held).

  Args:
    bits: bool[..., 20] cards held.

  Returns:
    int[..., 4].
  """
  b = np.asarray(bits).reshape(bits.shape[:-1] + (NUM_SUITS, NUM_RANKS))
  return (b * (1 << np.arange(NUM_RANKS))).sum(-1)


def class_key(masks, trump: int, trump_marked: bool) -> tuple[int, ...]:
  """A hand's class: its suit masks in canonical order (module doc).

  Args:
    masks: the four suit masks.
    trump: the trump suit.
    trump_marked: whether the trump suit comes first.

  Returns:
    The four masks, trump first (when marked), the rest descending.
  """
  masks = [int(m) for m in masks]
  if not trump_marked:
    return tuple(sorted(masks, reverse=True))
  rest = sorted((masks[s] for s in range(NUM_SUITS) if s != trump), reverse=True)
  return (masks[trump], *rest)


@functools.lru_cache(maxsize=None)
def hand_classes(trump_marked: bool) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
  """The classes and every hand's class at every trump.

  Args:
    trump_marked: whether the trump suit is marked (Schnapser, Bauern).

  Returns:
    (classes, ids): the classes in ascending key order, and int32[4, 15504],
    each hand's class index at each trump.
  """
  masks = suit_masks(HAND_BITS)
  keys = [
      [class_key(m, t, trump_marked) for m in masks] for t in range(NUM_SUITS)
  ]
  classes = tuple(sorted({k for row in keys for k in row}))
  index = {k: i for i, k in enumerate(classes)}
  ids = np.array([[index[k] for k in row] for row in keys], np.int32)
  return classes, ids


def class_counts(trump_marked: bool) -> np.ndarray:
  """How many hands fall in each class at a fixed trump.

  Args:
    trump_marked: whether the trump suit is marked.

  Returns:
    int[num_classes]; sums to 15504.
  """
  classes, ids = hand_classes(trump_marked)
  return np.bincount(ids[0], minlength=len(classes))


def class_hand(key) -> list[int]:
  """A hand of class `key` with the trump (when marked) in suit 0.

  Args:
    key: the class's four masks.

  Returns:
    The five cards, ascending.
  """
  return [
      s * NUM_RANKS + r
      for s in range(NUM_SUITS)
      for r in range(NUM_RANKS)
      if int(key[s]) >> r & 1
  ]


def trump_marked_of(contract: int) -> bool:
  """Whether `contract` plays with trump (so its classes mark the trump)."""
  return contract != schnaps.LOMT


def class_weights(
    p: np.ndarray,
    counts: np.ndarray,
    *,
    lo: float,
    hi: float,
    weak_share: float,
) -> np.ndarray:
  """Each class's weight from its estimated chance of success.

  1 when p >= hi; a constant c when lo <= p < hi, chosen so those weak hands
  are `weak_share` of the dealt hands (by count); 0 below lo.

  Args:
    p: float[num_classes] the chance of success.
    counts: int[num_classes] hands per class.
    lo: the lowest chance kept.
    hi: the main group's lowest chance.
    weak_share: the weak group's share of the dealt hands.

  Returns:
    float[num_classes].
  """
  p = np.asarray(p, np.float64)
  main = p >= hi
  weak = (p >= lo) & ~main
  w = main.astype(np.float64)
  m_main = float((counts * main).sum())
  m_weak = float((counts * weak).sum())
  if m_weak > 0 and m_main > 0:
    w[weak] = weak_share / (1.0 - weak_share) * m_main / m_weak
  elif m_weak > 0:
    w[weak] = 1.0
  return w


def load_hands(path: str = HANDS_FILE) -> tuple[dict[str, Any], str]:
  """The hands file and its sha256.

  Args:
    path: the file.

  Returns:
    (the parsed file, its sha256 hex digest).

  Raises:
    ValueError: a file of another schema.
  """
  with open(path, "rb") as f:
    raw = f.read()
  data = json.loads(raw)
  if data.get("schema") != HANDS_SCHEMA:
    raise ValueError(f"{path}: schema {data.get('schema')!r} != {HANDS_SCHEMA}")
  return data, hashlib.sha256(raw).hexdigest()


def hand_weights(entry: Mapping[str, Any], contract: int) -> np.ndarray:
  """W[seat, trump, hand] of one contract's entry in the hands file.

  Args:
    entry: the contract's entry ({"classes", "weight"}; weight per class per
      seat, or per class for one seat for all).
    contract: the contract.

  Returns:
    float32[4, 4, 15504].

  Raises:
    ValueError: the entry's classes are not this contract's.
  """
  marked = trump_marked_of(contract)
  classes, ids = hand_classes(marked)
  got = tuple(tuple(int(m) for m in k) for k in entry["classes"])
  if got != classes:
    raise ValueError(
        f"the hands file's {len(got)} classes are not the "
        f"{len(classes)} classes of contract {contract}"
    )
  w = np.asarray(entry["weight"], np.float64)
  if w.ndim == 1:
    w = np.repeat(w[:, None], NUM_PLAYERS, axis=1)
  if w.shape != (len(classes), NUM_PLAYERS) or (w < 0).any():
    raise ValueError(f"bad weights of shape {w.shape}")
  out = np.stack([w[ids[t]].T for t in range(NUM_SUITS)], axis=1)  # [4,4,H]
  if (out.sum(-1) <= 0).any():
    raise ValueError("a seat and trump with no hand of positive weight")
  return out.astype(np.float32)


# -- the game ------------------------------------------------------------------


def _i8(v) -> jax.Array:
  return jnp.asarray(v, jnp.int8)


def _i32(v) -> jax.Array:
  return jnp.asarray(v, jnp.int32)


class ForcedState(NamedTuple):
  """A forced deal: the scenario, then the schnaps state it replays into."""

  base: schnaps.GameState
  stage: jax.Array  # int8 (STAGE_*)
  declarer: jax.Array  # int8, -1 until drawn
  trump: jax.Array  # int8, -1 until drawn
  holder: jax.Array  # int8[20]: the seat dealt each card, -1 while undealt
  order: jax.Array  # int8[20]: the cards in deal order, -1 past n
  n: jax.Array  # int8: cards dealt
  replayed: jax.Array  # int8: base moves replayed


class SchnapsForced(game_lib.Game):
  """Schnaps with `contract` forced (module doc)."""

  def __init__(self, contract: int, weights: np.ndarray):
    if contract not in CONTRACTS.values():
      raise ValueError(f"contract {contract} is not a forced contract")
    w = np.asarray(weights, np.float32)
    if w.shape != (NUM_PLAYERS, NUM_SUITS, NUM_HANDS):
      raise ValueError(f"weights of shape {w.shape}")
    self.contract = int(contract)
    self.base = schnaps.Schnaps(max_flecks=MAX_FLECKS)
    self._weights = jnp.asarray(w)
    self._bits = jnp.asarray(HAND_BITS, jnp.float32)
    self._held = jnp.asarray(HAND_BITS, jnp.bool_)

  @property
  def num_actions(self) -> int:
    return self.base.num_actions

  @property
  def num_players(self) -> int:
    return self.base.num_players

  @property
  def obs_shape(self) -> tuple[int, ...]:
    return self.base.obs_shape

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    return self.base.info_state_shape

  @property
  def max_chance_outcomes(self) -> int:
    return NUM_CARDS

  @property
  def max_game_length(self) -> int:
    return 2 + NUM_CARDS + MAX_REPLAY + NUM_CARDS

  def new_initial_state(self) -> ForcedState:
    return ForcedState(
        base=self.base.new_initial_state(),
        stage=_i8(STAGE_SEAT),
        declarer=_i8(-1),
        trump=_i8(-1),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        order=jnp.full((NUM_CARDS,), -1, jnp.int8),
        n=_i8(0),
        replayed=_i8(0),
    )

  # -- the scenario ---------------------------------------------------------

  def _declarer_probs(self, s: ForcedState) -> jax.Array:
    """P(next declarer card) from the hand distribution (module doc)."""
    seat = jnp.maximum(_i32(s.declarer), 0)
    trump = jnp.maximum(_i32(s.trump), 0)
    w = self._weights[seat, trump]  # [H]
    dealt = (s.holder == s.declarer) & (s.declarer >= 0)
    held = jnp.all(self._held | ~dealt[None, :], axis=1)
    p = (w * held) @ self._bits  # [20]
    p = jnp.where(dealt | (s.holder >= 0), 0.0, p)
    total = p.sum()
    free = (s.holder < 0).astype(jnp.float32)
    return jnp.where(total > 0, p / jnp.maximum(total, 1e-30), free / free.sum())

  def _others(self, s: ForcedState) -> jax.Array:
    """The three other seats, ascending."""
    seats = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
    key = jnp.where(seats == _i32(s.declarer), NUM_PLAYERS, seats)
    return jnp.sort(key)[: NUM_PLAYERS - 1]

  def script(self, s: ForcedState) -> jax.Array:
    """The base moves the replay applies, int32[MAX_REPLAY] (-1 past the end).

    Args:
      s: a fully dealt forced state.

    Returns:
      The moves (module doc, step 4).
    """
    cards = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    pos = jnp.zeros((NUM_CARDS,), jnp.int32).at[_i32(s.order)].set(
        jnp.arange(NUM_CARDS, dtype=jnp.int32), mode="drop"
    )
    holder = _i32(s.holder)
    trump = _i32(s.trump)

    def seat_cards(seat, trumps_first):
      key = pos + jnp.where(
          trumps_first & (cards // NUM_RANKS != trump), NUM_CARDS, 0
      )
      key = jnp.where(holder == seat, key, 4 * NUM_CARDS)
      return jnp.argsort(key)[:HAND_SIZE].astype(jnp.int32)

    s0 = seat_cards(0, True)
    deal = jnp.concatenate([
        s0[:3],
        jnp.asarray([schnaps.FIRST_TRUMP_ACTION], jnp.int32) + trump[None],
        s0[3:],
        seat_cards(1, False),
        seat_cards(2, False),
        seat_cards(3, False),
    ])  # 21
    d = _i32(s.declarer)
    k = jnp.arange(MAX_REPLAY - deal.shape[0], dtype=jnp.int32)  # 7 bid slots
    call = schnaps.FIRST_CALL_ACTION + self.contract - 1
    bids = jnp.where(
        k < d,
        schnaps.PASS_ACTION,
        jnp.where(k == d, call, jnp.where(k <= d + 3, schnaps.PASS_ACTION, -1)),
    )
    return jnp.concatenate([deal, bids.astype(jnp.int32)])

  # -- Game -------------------------------------------------------------------

  def current_player(self, state: ForcedState) -> jax.Array:
    return jnp.where(
        state.stage < STAGE_PLAY,
        _i32(game_lib.CHANCE_PLAYER_ID),
        _i32(self.base.current_player(state.base)),
    )

  def is_terminal(self, state: ForcedState) -> jax.Array:
    return (state.stage == STAGE_PLAY) & self.base.is_terminal(state.base)

  def is_chance_node(self, state: ForcedState) -> jax.Array:
    return state.stage < STAGE_PLAY

  def chance_outcomes(self, state: ForcedState) -> jax.Array:
    st = state.stage
    four = jnp.where(jnp.arange(NUM_CARDS) < NUM_PLAYERS, 0.25, 0.0)
    free = (state.holder < 0).astype(jnp.float32)
    uniform = free / jnp.maximum(free.sum(), 1.0)
    first = jnp.zeros((NUM_CARDS,), jnp.float32).at[0].set(1.0)
    return jnp.select(
        [st <= STAGE_TRUMP, st == STAGE_DECLARER, st == STAGE_OTHERS],
        [four, self._declarer_probs(state), uniform],
        first,
    ).astype(jnp.float32)

  def apply_action(self, state: ForcedState, action: jax.Array) -> ForcedState:
    s = state
    a = _i32(action)
    st = s.stage
    n = _i32(s.n)
    # Stages 0 and 1: the declarer's seat, then the trump.
    seat_s = s._replace(declarer=_i8(a), stage=_i8(STAGE_TRUMP))
    trump_s = s._replace(trump=_i8(a), stage=_i8(STAGE_DECLARER))
    # Stages 2 and 3: a card to the declarer, then to the other seats.
    others = self._others(s)
    to = jnp.where(
        st == STAGE_DECLARER,
        _i32(s.declarer),
        others[jnp.clip((n - HAND_SIZE) // HAND_SIZE, 0, NUM_PLAYERS - 2)],
    )
    n1 = n + 1
    dealt = s._replace(
        holder=s.holder.at[a].set(_i8(to), mode="drop"),
        order=s.order.at[n].set(_i8(a), mode="drop"),
        n=_i8(n1),
        stage=_i8(
            jnp.where(
                n1 == NUM_CARDS,
                STAGE_REPLAY,
                jnp.where(n1 >= HAND_SIZE, STAGE_OTHERS, STAGE_DECLARER),
            )
        ),
    )
    # Stage 4: one scripted base move (the chance outcome is ignored).
    r = _i32(s.replayed)
    move = self.script(s)[jnp.clip(r, 0, MAX_REPLAY - 1)]
    base = self.base.apply_action(s.base, jnp.maximum(move, 0))
    done = base.phase == schnaps.PHASE_PLAY
    replay_s = s._replace(
        base=base,
        replayed=_i8(r + 1),
        stage=_i8(jnp.where(done, STAGE_PLAY, STAGE_REPLAY)),
    )
    # Play: schnaps's.
    play_s = s._replace(base=self.base.apply_action(s.base, a))
    out = play_s
    for cond, alt in (
        (st == STAGE_REPLAY, replay_s),
        ((st == STAGE_DECLARER) | (st == STAGE_OTHERS), dealt),
        (st == STAGE_TRUMP, trump_s),
        (st == STAGE_SEAT, seat_s),
    ):
      out = jax.tree_util.tree_map(
          lambda x, y, c=cond: jnp.where(c, x, y), alt, out
      )
    return out

  def legal_actions_mask(self, state: ForcedState) -> jax.Array:
    return jnp.where(
        state.stage == STAGE_PLAY,
        self.base.legal_actions_mask(state.base),
        False,
    )

  def returns(self, state: ForcedState) -> jax.Array:
    return self.base.returns(state.base)

  def information_state_tensor(self, state: ForcedState, player) -> jax.Array:
    return self.base.information_state_tensor(state.base, player)

  def observation_tensor(self, state: ForcedState, player) -> jax.Array:
    return self.base.observation_tensor(state.base, player)


# -- the envs ------------------------------------------------------------------


def rules_sha(contract_name: str, hands_sha: str) -> str:
  """The forced env's rules identity (module doc, RULES IDENTITY).

  Args:
    contract_name: the env suffix ("schnapser", "lomt", "bauern").
    hands_sha: the hands file's sha256.

  Returns:
    The sha256 hex digest.
  """
  rules = {
      "schema": "schnaps-forced-rules/1",
      "schnaps": schnaps_env.rules_sha({"max_flecks": MAX_FLECKS}),
      "contract": contract_name,
      "hands_sha256": hands_sha,
  }
  return hashlib.sha256(
      json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


class SchnapsForcedEnv(schnaps_env.SchnapsEnv):
  """Schnaps with one contract forced; a subclass per contract (module doc).

  Attributes:
    contract_name: the env suffix ("schnapser", "lomt", "bauern").
    hands_sha: the hands file's sha256.
  """

  contract_name: str = ""
  max_len = NUM_CARDS  # the plays only

  def __init__(self, hands_path: Optional[str] = None):  # pylint: disable=super-init-not-called
    if self.contract_name not in CONTRACTS:
      raise TypeError(f"{type(self).__name__}: set contract_name")
    self.options = {}
    self.max_len = NUM_CARDS
    data, self.hands_sha = load_hands(hands_path or HANDS_FILE)
    contract = CONTRACTS[self.contract_name]
    weights = hand_weights(data["contracts"][self.contract_name], contract)
    jax_game_env.JaxGameEnv.__init__(self, SchnapsForced(contract, weights))
    assert self.num_players == NUM_PLAYERS
    assert self.num_actions == schnaps_env.NUM_ACTIONS

  @property
  def declared_rules_sha(self) -> str:
    return rules_sha(self.contract_name, self.hands_sha)

  def terminal(self, s: ForcedState) -> jax.Array:
    return jnp.asarray(self.game.is_terminal(s), jnp.bool_)

  def info_state(self, s: ForcedState, seat) -> dict[str, jax.Array]:
    return super().info_state(s.base, seat)

  def labels(self, s: ForcedState, seat) -> Mapping[str, jax.Array]:
    return super().labels(s.base, seat)


class SchnapsForcedSchnapserEnv(SchnapsForcedEnv):
  name = "schnaps_forced_schnapser"
  contract_name = "schnapser"


class SchnapsForcedLomtEnv(SchnapsForcedEnv):
  name = "schnaps_forced_lomt"
  contract_name = "lomt"


class SchnapsForcedBauernEnv(SchnapsForcedEnv):
  name = "schnaps_forced_bauern"
  contract_name = "bauern"


def hide_hidden_cards(env, s: ForcedState, seat, key) -> ForcedState:
  """Privacy hook: schnaps's, on the replayed state (envs.schnaps).

  Args:
    env: unused.
    s: the state.
    seat: the seat whose view is kept.
    key: the shuffle's key.

  Returns:
    The state with the base state's hidden cards re-dealt.
  """
  return s._replace(base=schnaps_env.hide_hidden_cards(env, s.base, seat, key))
