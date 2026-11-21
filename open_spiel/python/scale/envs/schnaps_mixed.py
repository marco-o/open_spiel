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

"""Schnaps with a mix of deals: Normal games and forced contracts, one env.

Step 2 of Docker/gpu-training.md: one network that plays every contract.
Each deal starts with a chance node that picks its scenario (MIX):

  * **Normal** (scenario 0): a Normal deal as in schnaps_normal. Chance deals
    the Caller's three cards, the Caller CHOOSES trump (a decision), chance
    deals the other seventeen, then four passes are replayed (chance nodes
    with one outcome) and play starts. No call and no fleck is ever made,
    so the play is a Normal game.
  * **Schnapser, Lomt, Bauern** (scenarios 1..3, the contract ids): a forced
    deal exactly as envs.schnaps_forced deals it: chance picks the
    declarer's seat and the trump, deals the declarer a plausible hand from
    the hands file, deals the rest uniformly, then the deal and the bids are
    replayed into the engine.

Every scenario reaches a genuine schnaps(max_flecks=2) state, so the
InfoState and the encoders are schnaps's, and Elena and the card planes read
it unchanged. The decisions are the trump choice (Normal deals) and the card
plays and declarations. Bids are never decisions here: the value-head bidder
(envs.schnaps_bidding) bids.

RULES IDENTITY. ``declared_rules_sha``: the sha256 over canonical JSON of
schnaps's rules identity at max_flecks 2, the mix and the hands file's
sha256.

Not served (envs.registry): a network trained here has never bid.
"""

import hashlib
import json
from typing import Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import schnaps
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.envs import schnaps as schnaps_env
from open_spiel.python.scale.envs import schnaps_forced as sf

NUM_CARDS = schnaps.NUM_CARDS
NUM_PLAYERS = schnaps.NUM_PLAYERS
NUM_SUITS = sf.NUM_SUITS
HAND_SIZE = sf.HAND_SIZE
MAX_FLECKS = sf.MAX_FLECKS

#: The scenarios, by name; the index is the contract id (0 Normal ..).
SCENARIOS = ("normal", "schnapser", "lomt", "bauern")
NUM_SCENARIOS = len(SCENARIOS)
assert tuple(sf.CONTRACTS[n] for n in SCENARIOS[1:]) == (1, 2, 3)
#: Each scenario's share of the deals (Docker/gpu-training.md, step 2a).
MIX = (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)

# The stages. The forced stages follow schnaps_forced's order; STAGE_PLAY is
# last, so stage < STAGE_PLAY is "before play".
(
    STAGE_SCENARIO,
    STAGE_SEAT,
    STAGE_TRUMP,
    STAGE_DECLARER,
    STAGE_OTHERS,
    STAGE_REPLAY,
    STAGE_NORMAL_DEAL,
    STAGE_NORMAL_BIDS,
    STAGE_PLAY,
) = range(9)


def _i8(v) -> jax.Array:
  return jnp.asarray(v, jnp.int8)


def _i32(v) -> jax.Array:
  return jnp.asarray(v, jnp.int32)


class MixedState(NamedTuple):
  """A mixed deal: the scenario, its forced-deal fields, the engine state."""

  base: schnaps.GameState
  stage: jax.Array  # int8 (STAGE_*)
  scenario: jax.Array  # int8, -1 until drawn (0 Normal, else the contract)
  declarer: jax.Array  # int8, forced deals: -1 until drawn
  trump: jax.Array  # int8, forced deals: -1 until drawn
  holder: jax.Array  # int8[20], forced deals: the seat dealt each card
  order: jax.Array  # int8[20], forced deals: the cards in deal order
  n: jax.Array  # int8, forced deals: cards dealt
  replayed: jax.Array  # int8, forced deals: base moves replayed


class SchnapsMixed(game_lib.Game):
  """Schnaps with the scenario mix `mix` (module doc)."""

  def __init__(self, mix, weights: np.ndarray):
    """Builds the game.

    Args:
      mix: float[4], each scenario's share (sums to 1).
      weights: float32[3, 4, 4, 15504], W[seat, trump, hand] of Schnapser,
        Lomt and Bauern (schnaps_forced.hand_weights).

    Raises:
      ValueError: a bad mix or weights.
    """
    m = np.asarray(mix, np.float64)
    if m.shape != (NUM_SCENARIOS,) or (m < 0).any() or abs(m.sum() - 1) > 1e-6:
      raise ValueError(f"bad scenario mix {mix!r}")
    w = np.asarray(weights, np.float32)
    if w.shape != (NUM_SCENARIOS - 1, NUM_PLAYERS, NUM_SUITS, sf.NUM_HANDS):
      raise ValueError(f"weights of shape {w.shape}")
    self.mix = tuple(float(x) for x in m)
    self.base = schnaps.Schnaps(max_flecks=MAX_FLECKS)
    self._mix = jnp.zeros((NUM_CARDS,), jnp.float32).at[:NUM_SCENARIOS].set(
        jnp.asarray(m, jnp.float32)
    )
    self._weights = jnp.asarray(w)
    self._bits = jnp.asarray(sf.HAND_BITS, jnp.float32)
    self._held = jnp.asarray(sf.HAND_BITS, jnp.bool_)

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
    # The scenario, then the longer setup (a forced deal's), then the plays.
    return 1 + 2 + NUM_CARDS + sf.MAX_REPLAY + NUM_CARDS

  def new_initial_state(self) -> MixedState:
    return MixedState(
        base=self.base.new_initial_state(),
        stage=_i8(STAGE_SCENARIO),
        scenario=_i8(-1),
        declarer=_i8(-1),
        trump=_i8(-1),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        order=jnp.full((NUM_CARDS,), -1, jnp.int8),
        n=_i8(0),
        replayed=_i8(0),
    )

  # -- the forced scenarios (schnaps_forced's, per contract) ----------------

  def _declarer_probs(self, s: MixedState) -> jax.Array:
    """P(next declarer card) from the scenario's hand distribution."""
    c = jnp.clip(_i32(s.scenario) - 1, 0, NUM_SCENARIOS - 2)
    seat = jnp.maximum(_i32(s.declarer), 0)
    trump = jnp.maximum(_i32(s.trump), 0)
    w = self._weights[c, seat, trump]  # [H]
    dealt = (s.holder == s.declarer) & (s.declarer >= 0)
    held = jnp.all(self._held | ~dealt[None, :], axis=1)
    p = (w * held) @ self._bits  # [20]
    p = jnp.where(dealt | (s.holder >= 0), 0.0, p)
    total = p.sum()
    free = (s.holder < 0).astype(jnp.float32)
    return jnp.where(total > 0, p / jnp.maximum(total, 1e-30), free / free.sum())

  def _others(self, s: MixedState) -> jax.Array:
    """The three seats other than the declarer, ascending."""
    seats = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
    key = jnp.where(seats == _i32(s.declarer), NUM_PLAYERS, seats)
    return jnp.sort(key)[: NUM_PLAYERS - 1]

  def script(self, s: MixedState) -> jax.Array:
    """A forced deal's replayed base moves, int32[MAX_REPLAY] (-1 past the end).

    schnaps_forced.SchnapsForced.script with the scenario's contract.

    Args:
      s: a fully dealt forced state.

    Returns:
      The moves.
    """
    cards = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    pos = jnp.zeros((NUM_CARDS,), jnp.int32).at[_i32(s.order)].set(
        jnp.arange(NUM_CARDS, dtype=jnp.int32), mode="drop"
    )
    holder = _i32(s.holder)
    trump = _i32(s.trump)

    def seat_cards(seat, trumps_first):
      key = pos + jnp.where(
          trumps_first & (cards // sf.NUM_RANKS != trump), NUM_CARDS, 0
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
    k = jnp.arange(sf.MAX_REPLAY - deal.shape[0], dtype=jnp.int32)
    call = schnaps.FIRST_CALL_ACTION + _i32(s.scenario) - 1
    bids = jnp.where(
        k < d,
        schnaps.PASS_ACTION,
        jnp.where(k == d, call, jnp.where(k <= d + 3, schnaps.PASS_ACTION, -1)),
    )
    return jnp.concatenate([deal, bids.astype(jnp.int32)])

  # -- Game -------------------------------------------------------------------

  def _base_driven(self, state: MixedState) -> jax.Array:
    """Whether the engine itself drives this node (Normal deal, or play)."""
    return (state.stage == STAGE_NORMAL_DEAL) | (state.stage == STAGE_PLAY)

  def current_player(self, state: MixedState) -> jax.Array:
    return jnp.where(
        self._base_driven(state),
        _i32(self.base.current_player(state.base)),
        _i32(game_lib.CHANCE_PLAYER_ID),
    )

  def is_terminal(self, state: MixedState) -> jax.Array:
    return (state.stage == STAGE_PLAY) & self.base.is_terminal(state.base)

  def is_chance_node(self, state: MixedState) -> jax.Array:
    return jnp.where(
        self._base_driven(state),
        self.base.is_chance_node(state.base),
        True,
    )

  def chance_outcomes(self, state: MixedState) -> jax.Array:
    st = state.stage
    four = jnp.where(jnp.arange(NUM_CARDS) < NUM_PLAYERS, 0.25, 0.0)
    free = (state.holder < 0).astype(jnp.float32)
    uniform = free / jnp.maximum(free.sum(), 1.0)
    first = jnp.zeros((NUM_CARDS,), jnp.float32).at[0].set(1.0)
    base = self.base.chance_outcomes(state.base).astype(jnp.float32)
    return jnp.select(
        [
            st == STAGE_SCENARIO,
            (st == STAGE_SEAT) | (st == STAGE_TRUMP),
            st == STAGE_DECLARER,
            st == STAGE_OTHERS,
            st == STAGE_NORMAL_DEAL,
        ],
        [self._mix, four, self._declarer_probs(state), uniform, base],
        first,
    ).astype(jnp.float32)

  def apply_action(self, state: MixedState, action: jax.Array) -> MixedState:
    s = state
    a = _i32(action)
    st = s.stage
    n = _i32(s.n)
    # The scenario.
    scenario_s = s._replace(
        scenario=_i8(a),
        stage=_i8(jnp.where(a == schnaps.NORMAL, STAGE_NORMAL_DEAL, STAGE_SEAT)),
    )
    # Forced: the declarer's seat, then the trump.
    seat_s = s._replace(declarer=_i8(a), stage=_i8(STAGE_TRUMP))
    trump_s = s._replace(trump=_i8(a), stage=_i8(STAGE_DECLARER))
    # Forced: a card to the declarer, then to the other seats.
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
    # One engine move: a forced replay's scripted move, a Normal deal's
    # replayed pass, or the action itself (Normal deal and play).
    r = _i32(s.replayed)
    scripted = self.script(s)[jnp.clip(r, 0, sf.MAX_REPLAY - 1)]
    move = jnp.select(
        [st == STAGE_REPLAY, st == STAGE_NORMAL_BIDS],
        [jnp.maximum(scripted, 0), _i32(schnaps.PASS_ACTION)],
        a,
    )
    base = self.base.apply_action(s.base, move)
    to_play = base.phase == schnaps.PHASE_PLAY
    replay_s = s._replace(
        base=base,
        replayed=_i8(r + 1),
        stage=_i8(jnp.where(to_play, STAGE_PLAY, STAGE_REPLAY)),
    )
    normal_deal_s = s._replace(
        base=base,
        stage=_i8(
            jnp.where(
                base.phase == schnaps.PHASE_BIDDING,
                STAGE_NORMAL_BIDS,
                STAGE_NORMAL_DEAL,
            )
        ),
    )
    normal_bids_s = s._replace(
        base=base,
        stage=_i8(jnp.where(to_play, STAGE_PLAY, STAGE_NORMAL_BIDS)),
    )
    out = s._replace(base=base)  # play
    for cond, alt in (
        (st == STAGE_NORMAL_BIDS, normal_bids_s),
        (st == STAGE_NORMAL_DEAL, normal_deal_s),
        (st == STAGE_REPLAY, replay_s),
        ((st == STAGE_DECLARER) | (st == STAGE_OTHERS), dealt),
        (st == STAGE_TRUMP, trump_s),
        (st == STAGE_SEAT, seat_s),
        (st == STAGE_SCENARIO, scenario_s),
    ):
      out = jax.tree_util.tree_map(
          lambda x, y, c=cond: jnp.where(c, x, y), alt, out
      )
    return out

  def legal_actions_mask(self, state: MixedState) -> jax.Array:
    decides = self._base_driven(state) & ~self.base.is_chance_node(state.base)
    return jnp.where(decides, self.base.legal_actions_mask(state.base), False)

  def returns(self, state: MixedState) -> jax.Array:
    return self.base.returns(state.base)

  def information_state_tensor(self, state: MixedState, player) -> jax.Array:
    return self.base.information_state_tensor(state.base, player)

  def observation_tensor(self, state: MixedState, player) -> jax.Array:
    return self.base.observation_tensor(state.base, player)


# -- the env -------------------------------------------------------------------


def rules_sha(mix, hands_sha: str) -> str:
  """The mixed env's rules identity (module doc, RULES IDENTITY).

  Args:
    mix: the scenario mix.
    hands_sha: the hands file's sha256.

  Returns:
    The sha256 hex digest.
  """
  rules = {
      "schema": "schnaps-mixed-rules/1",
      "schnaps": schnaps_env.rules_sha({"max_flecks": MAX_FLECKS}),
      "scenarios": list(SCENARIOS),
      "mix": [round(float(x), 9) for x in mix],
      "hands_sha256": hands_sha,
  }
  return hashlib.sha256(
      json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def load_weights(path: Optional[str] = None) -> tuple[np.ndarray, str]:
  """The forced contracts' hand weights and the hands file's sha256.

  Args:
    path: the hands file (default schnaps_forced.HANDS_FILE).

  Returns:
    (float32[3, 4, 4, 15504] in SCENARIOS[1:] order, the sha256).
  """
  data, sha = sf.load_hands(path or sf.HANDS_FILE)
  w = np.stack([
      sf.hand_weights(data["contracts"][name], sf.CONTRACTS[name])
      for name in SCENARIOS[1:]
  ])
  return w, sha


class SchnapsMixedEnv(schnaps_env.SchnapsEnv):
  """Normal deals and forced contracts in one env (module doc).

  Attributes:
    mix: the scenario mix.
    hands_sha: the hands file's sha256.
  """

  name = "schnaps_mixed"
  #: The decisions: the trump choice (Normal deals), then the plays.
  max_len = 1 + NUM_CARDS

  def __init__(self, hands_path: Optional[str] = None, mix=MIX):  # pylint: disable=super-init-not-called
    self.options = {}
    self.max_len = 1 + NUM_CARDS
    weights, self.hands_sha = load_weights(hands_path)
    jax_game_env.JaxGameEnv.__init__(self, SchnapsMixed(mix, weights))
    self.mix = self.game.mix
    assert self.num_players == NUM_PLAYERS
    assert self.num_actions == schnaps_env.NUM_ACTIONS

  @property
  def declared_rules_sha(self) -> str:
    return rules_sha(self.mix, self.hands_sha)

  def terminal(self, s: MixedState) -> jax.Array:
    return jnp.asarray(self.game.is_terminal(s), jnp.bool_)

  def info_state(self, s: MixedState, seat) -> dict[str, jax.Array]:
    return super().info_state(s.base, seat)

  def labels(self, s: MixedState, seat) -> Mapping[str, jax.Array]:
    return super().labels(s.base, seat)


def hide_hidden_cards(env, s: MixedState, seat, key) -> MixedState:
  """Privacy hook: schnaps's, on the engine state (envs.schnaps).

  Args:
    env: unused.
    s: the state.
    seat: the seat whose view is kept.
    key: the shuffle's key.

  Returns:
    The state with the engine state's hidden cards re-dealt.
  """
  return s._replace(base=schnaps_env.hide_hidden_cards(env, s.base, seat, key))
