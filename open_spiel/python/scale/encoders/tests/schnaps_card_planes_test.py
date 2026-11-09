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

"""Tests of encoders.schnaps_card_planes: shape, privacy, planes, value rows."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.jax.games import schnaps as game
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.encoders import schnaps_card_planes as scp
from open_spiel.python.scale.envs import schnaps

# pylint: enable=g-import-not-at-top,wrong-import-position

ENV = schnaps.SchnapsEnv()
_PLANES = jax.jit(jax.vmap(scp.card_planes))
_WAIT = jax.jit(jax.vmap(scp.waiting_row))
_INFO4 = jax.jit(
    lambda s: jax.vmap(lambda q: ENV.info_state(s, q))(jnp.arange(4))
)
CH = {c: i for i, c in enumerate(scp.CHANNELS)}
_PHASE_PLANE = {
    game.PHASE_CHOOSE_TRUMP: "b_phase_trump",
    game.PHASE_BIDDING: "b_phase_bidding",
    game.PHASE_PLAY: "b_phase_play",
}


def _states(seed, n_games=4, call_bias=0.0):
  """Every player node of n_games random deals (uniform legal moves).

  Args:
    seed: the numpy seed.
    n_games: deals to play.
    call_bias: the chance a bid is a call or a fleck when one is legal.

  Yields:
    (state, actor, action) at each player node, then (state, -1, -1) at
    each terminal.
  """
  x64.enable_x64()
  g = ENV.game
  rng = np.random.default_rng(seed)
  apply = jax.jit(g.apply_action)
  legal = jax.jit(g.legal_actions_mask)
  chance = jax.jit(g.chance_outcomes)
  for _ in range(n_games):
    s = g.new_initial_state()
    while not bool(g.is_terminal(s)):
      if bool(g.is_chance_node(s)):
        p = np.asarray(chance(s), np.float64)
        a = int(rng.choice(len(p), p=p / p.sum()))
      else:
        acts = np.flatnonzero(np.asarray(legal(s)))
        calls = acts[acts >= game.FLECK_ACTION]
        if len(calls) and rng.random() < call_bias:
          a = int(rng.choice(calls))
        else:
          a = int(rng.choice(acts))
        yield s, int(g.current_player(s)), a
      s = apply(s, jnp.asarray(a, jnp.int32))
    yield s, -1, -1


def test_registered_shape_and_finite():
  enc = registry.get(scp.NAME)
  assert enc.shape == (70, 20) and enc.game == "schnaps"
  n = 0
  for s, _, _ in _states(0, 1):
    x = np.asarray(_PLANES(_INFO4(s)))
    assert x.shape == (4, 70, 20) and np.isfinite(x).all()
    n += 1
  assert n >= 7  # the trump choice, four passes or more, the terminal


@pytest.mark.parametrize("call_bias", [0.0, 0.5])
def test_decision_planes(call_bias):
  """The seat to move sees itself on move, in the engine's phase, its hand.

  Args:
    call_bias: the random games' call bias.
  """
  for s, actor, _ in _states(1, 6, call_bias):
    x = np.asarray(_PLANES(_INFO4(s)))
    if actor < 0:
      assert (x[:, CH["b_terminal"]] == 1).all()
      assert (x[:, [CH[f"b_actor_rel{k}"] for k in range(4)]] == 0).all()
      continue
    xa = x[actor]
    assert xa[CH["b_actor_rel0"]].min() == 1.0, "the actor is the viewer"
    assert xa[CH[_PHASE_PLANE[int(s.phase)]], 0] == 1.0
    hand = np.asarray(ENV.info_state(s, actor)["hand"])
    np.testing.assert_array_equal(xa[CH["my_hand"]], hand.astype(np.float32))
    # The cards the actor may play are in its hand.
    mask = np.asarray(ENV.legal_mask(s))[: game.NUM_CARDS]
    assert not (mask & ~hand).any()
    for c in ("b_trump0", "b_trump_active", "b_contract0", "b_trick_num_over5"):
      assert (x[:, CH[c]] == x[0, CH[c]]).all(), c


def test_privacy_hidden_cards_do_not_change_a_seats_planes():
  """Re-dealing what a seat cannot see leaves its planes; others move."""
  hide = jax.jit(lambda s, q, k: schnaps.hide_hidden_cards(ENV, s, q, k))
  changed_other = 0
  for i, (s, actor, _) in enumerate(_states(2, 3)):
    if actor < 0 or i % 2:
      continue
    for q in range(4):
      t = hide(s, jnp.int32(q), jax.random.key(100 * i + q))
      a = np.asarray(scp.card_planes(ENV.info_state(s, q)))
      b = np.asarray(scp.card_planes(ENV.info_state(t, q)))
      np.testing.assert_array_equal(a, b)
      assert bool(scp.waiting_row(ENV.info_state(s, q))) == bool(
          scp.waiting_row(ENV.info_state(t, q))
      )
      o = (q + 1) % 4
      if not np.array_equal(
          np.asarray(scp.card_planes(ENV.info_state(s, o))),
          np.asarray(scp.card_planes(ENV.info_state(t, o))),
      ):
        changed_other += 1
  assert changed_other > 0


def test_voids_are_true():
  """A seat marked void in a suit holds no card of it."""
  marked = 0
  for s, actor, _ in _states(4, 6):
    if actor < 0:
      continue
    x = np.asarray(_PLANES(_INFO4(s)))
    holder = np.asarray(s.holder)
    played = np.asarray(s.played)
    for viewer in range(4):
      for k in (1, 2, 3):
        seat = (viewer + k) % 4
        void = x[viewer, CH[f"void_rel{k}"]] > 0
        held = (holder == seat) & ~played
        assert not (void & held).any(), (viewer, k)
        marked += int(void.any())
  assert marked > 0


@pytest.mark.parametrize("call_bias", [0.0, 0.5])
def test_waiting_row_matches_a_round_robin_clock(call_bias):
  """waiting_row marks exactly the seats a row clock passes while waiting.

  The clock starts on the first decision of each deal, the Caller's trump
  choice (out of scope); each row belongs to seat clock % 4: the seat on
  move acts, any other seat waits (a value-only row) at the state before
  the next move.

  Args:
    call_bias: the random games' call bias.
  """
  n_rows = 0
  clock = None
  for s, actor, _ in _states(3, 8, call_bias):
    if actor < 0:
      clock = None
      continue
    want = np.zeros(4, bool)
    first = clock is None
    if first:
      clock = actor
    while clock % 4 != actor:
      want[clock % 4] = True
      clock += 1
    clock += 1  # the actor's row
    got = np.asarray(_WAIT(_INFO4(s)))
    np.testing.assert_array_equal(
        got,
        want,
        err_msg=f"phase {int(s.phase)} n_played {int(s.n_played)}",
    )
    assert not got[actor]
    n_rows += int(want.sum())
  assert n_rows > 20


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
