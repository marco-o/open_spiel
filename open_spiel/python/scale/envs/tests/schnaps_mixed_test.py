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

"""envs.schnaps_mixed: Normal deals and forced contracts in one env."""

import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as G
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import schnaps_elena as elena
from open_spiel.python.scale.envs import schnaps_forced as sf
from open_spiel.python.scale.envs import schnaps_mixed as sm
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_env

# pylint: enable=g-import-not-at-top,wrong-import-position

NAME = "schnaps_mixed"


@pytest.fixture(scope="module", autouse=True)
def _x64():
  x64.enable_x64()


@pytest.fixture(scope="module")
def env():
  return env_registry.make(NAME)


def _deal(env, n, seed=0):
  keys = jax.random.split(jax.random.key(seed), n)
  return jax.jit(jax.vmap(env.init))(keys)


def test_the_scenarios_follow_the_mix(env):
  n = 4000
  s = _deal(env, n, seed=1)
  got = np.bincount(np.asarray(s.scenario), minlength=sm.NUM_SCENARIOS) / n
  want = np.asarray(sm.MIX)
  assert (np.abs(got - want) < 4 * np.sqrt(want * (1 - want) / n)).all(), got


def test_normal_deals_start_at_the_callers_trump_choice(env):
  s = _deal(env, 256, seed=2)
  normal = np.asarray(s.scenario) == G.NORMAL
  assert normal.any()
  stage = np.asarray(s.stage)
  phase = np.asarray(s.base.phase)
  assert (stage[normal] == sm.STAGE_NORMAL_DEAL).all()
  assert (phase[normal] == G.PHASE_CHOOSE_TRUMP).all()
  assert (np.asarray(s.base.current_player)[normal] == 0).all()
  legal = np.asarray(jax.vmap(env.legal_mask)(s))
  trumps = np.zeros(G.NUM_DISTINCT_ACTIONS, bool)
  trumps[G.FIRST_TRUMP_ACTION : G.FIRST_TRUMP_ACTION + 4] = True
  assert (legal[normal] == trumps).all()
  # The Caller holds exactly its first three cards.
  holder = np.asarray(s.base.holder)[normal]
  assert ((holder == 0).sum(-1) == 3).all() and ((holder >= 0).sum(-1) == 3).all()


def test_a_normal_deal_plays_a_normal_game_after_four_passes(env):
  s = _deal(env, 256, seed=3)
  normal = np.flatnonzero(np.asarray(s.scenario) == G.NORMAL)
  step = jax.jit(jax.vmap(env.step))
  keys = jax.random.split(jax.random.key(4), 256)
  trump = jnp.full((256,), G.FIRST_TRUMP_ACTION + 2, jnp.int32)
  s1 = step(s, trump, keys)
  b = s1.base
  assert (np.asarray(s1.stage)[normal] == sm.STAGE_PLAY).all()
  assert (np.asarray(b.phase)[normal] == G.PHASE_PLAY).all()
  assert (np.asarray(b.contract)[normal] == G.NORMAL).all()
  assert (np.asarray(b.trump)[normal] == 2).all()
  assert (np.asarray(b.flecks)[normal] == 0).all()
  assert not np.asarray(b.any_call)[normal].any()
  assert (np.asarray(b.n_bids)[normal] == 4).all()
  assert (np.asarray(b.bids)[normal, :4] == 0).all()
  holder = np.asarray(b.holder)[normal]
  for h in holder:
    assert np.bincount(h, minlength=4).tolist() == [5, 5, 5, 5]


def test_forced_deals_start_at_the_first_play_of_their_contract(env):
  s = _deal(env, 512, seed=5)
  sc = np.asarray(s.scenario)
  b = s.base
  for c in (G.SCHNAPSER, G.LOMT, G.BAUERN):
    rows = np.flatnonzero(sc == c)
    assert len(rows) > 20
    d = np.asarray(s.declarer)[rows].astype(np.int32)
    assert (np.asarray(s.stage)[rows] == sm.STAGE_PLAY).all()
    assert (np.asarray(b.phase)[rows] == G.PHASE_PLAY).all()
    assert (np.asarray(b.contract)[rows] == c).all()
    assert (np.asarray(b.declarer)[rows] == d).all()
    assert (np.asarray(b.trump)[rows] == np.asarray(s.trump)[rows]).all()
    for i, r in enumerate(rows):
      bids = np.asarray(b.bids[r]).tolist()
      want = [0] * d[i] + [c + 1] + [0, 0, 0]
      assert bids[: len(want)] == want


def _walk(env, seed):
  """One deal from the root, at random; returns (end state, base moves)."""
  rng = np.random.default_rng(seed)
  s = env.root()
  moves = []
  apply = jax.jit(env.apply)
  apply_chance = jax.jit(env.apply_chance)
  probs = jax.jit(env.chance_probs)
  legal = jax.jit(env.legal_mask)
  term = jax.jit(env.terminal)
  is_chance = jax.jit(env.is_chance)
  script = jax.jit(env.game.script)
  while not bool(term(s)):
    st = int(s.stage)
    if bool(is_chance(s)):
      p = np.asarray(probs(s), np.float64)
      o = int(rng.choice(len(p), p=p / p.sum()))
      if st == sm.STAGE_REPLAY:
        moves.append(int(np.asarray(script(s))[int(s.replayed)]))
      elif st == sm.STAGE_NORMAL_BIDS:
        moves.append(G.PASS_ACTION)
      elif st in (sm.STAGE_NORMAL_DEAL, sm.STAGE_PLAY):
        moves.append(o)
      s = apply_chance(s, jnp.int32(o))
    else:
      a = int(rng.choice(np.flatnonzero(np.asarray(legal(s)))))
      moves.append(a)
      s = apply(s, jnp.int32(a))
  return s, moves


def test_mixed_deals_are_schnaps_deals_in_pyspiel(env):
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game(f"schnaps(max_flecks={sm.MAX_FLECKS})")
  base_game = G.Schnaps(max_flecks=sm.MAX_FLECKS)
  seen = set()
  for seed in range(40):
    end, moves = _walk(env, seed)
    seen.add(int(end.scenario))
    st = game.new_initial_state()
    for a in moves:
      st.apply_action(a)
    assert st.is_terminal()
    assert np.allclose(st.returns(), np.asarray(env.returns(end)))
    for x, y in zip(
        jax.tree_util.tree_leaves(base_game.state_from_os(st)),
        jax.tree_util.tree_leaves(end.base),
    ):
      assert np.array_equal(np.asarray(x), np.asarray(y))
  assert seen == {0, 1, 2, 3}


def test_check_env_passes_with_privacy(env):
  check_env.check_env(env, hide_fn=sm.hide_hidden_cards, num_playouts=16)


def test_elena_plays_the_mixed_deals(env):
  ev = elena.evaluator(env, name=elena.MIXED_EVAL_NAME)
  unif = match.Entrant(
      "uniform",
      uniform.UniformPolicy(env.name, int(env.num_actions), int(env.num_players)),
      {},
      types.SAMPLE,
  )
  res = ev.head_to_head(elena.entrant(), unif, 16, jax.random.key(0))
  assert res.mean.mean > 0  # Elena beats random play


def test_a_bad_mix_is_refused():
  w, _ = sm.load_weights()
  with pytest.raises(ValueError, match="mix"):
    sm.SchnapsMixed((0.5, 0.5, 0.5, 0.0), w)
  with pytest.raises(ValueError, match="weights"):
    sm.SchnapsMixed(sm.MIX, w[:2])


def test_the_rules_identity_names_the_mix_and_hands(env):
  _, sha = sf.load_hands()
  assert env.declared_rules_sha == sm.rules_sha(sm.MIX, sha)
  assert sm.rules_sha((1.0, 0.0, 0.0, 0.0), sha) != env.declared_rules_sha
