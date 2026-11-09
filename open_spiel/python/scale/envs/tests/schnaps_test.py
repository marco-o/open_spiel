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

"""envs.schnaps and encoders.schnaps.

The Env contract with privacy, sides and identity, and equality with
OpenSpiel's schnaps on random games: every chance distribution, actor,
legal-action set and return equal, the deal replayed from pyspiel's chance
outcomes. SCHNAPS_PYSPIEL_GAMES sets the games per option set (default 40).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import schnaps as game_lib
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.encoders import schnaps as schnaps_enc
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.testing import check_env

pyspiel = pytest.importorskip("pyspiel")

#: Random games per option set of the pyspiel comparison.
PYSPIEL_GAMES = int(os.environ.get("SCHNAPS_PYSPIEL_GAMES", "40"))
OPTION_SETS = [{}, {"max_flecks": 0}, {"max_flecks": 4}]


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The default schnaps env.

  Returns:
    The SchnapsEnv.
  """
  return schnaps.SchnapsEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy."""
  check_env.check_env(env, hide_fn=schnaps.hide_hidden_cards, num_playouts=32)


def test_check_env_passes_without_flecks():
  """Check env passes at max_flecks=0."""
  check_env.check_env(
      schnaps.SchnapsEnv(max_flecks=0),
      hide_fn=schnaps.hide_hidden_cards,
      num_playouts=16,
  )


def test_sides_are_two_zero_sum_partnerships(env):
  """Sides are seats 0 and 2 against 1 and 3, zero-sum."""
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(
      4, 2, constant_sum=True
  )
  assert env.zero_sum
  assert sides_lib.sides_of(env).seats(0) == (0, 2)


def test_registry_spec_identity_and_encoder(env):
  """Registry spec identity and encoder."""
  assert registry.spec("schnaps").kwargs == schnaps.OPTIONS
  assert registry.spec_of(env) == registry.spec("schnaps")
  made = registry.make("schnaps")
  assert isinstance(made, schnaps.SchnapsEnv)
  sha = registry.rules_sha(made)
  assert sha == schnaps.rules_sha(schnaps.OPTIONS)
  assert registry.rules_sha(registry.make("schnaps", max_flecks=3)) != sha
  with pytest.raises(TypeError):
    registry.make("schnaps", max_flecks=True)
  with pytest.raises(ValueError):
    registry.make("schnaps", max_flecks=7)
  assert made.max_len == 1 + game_lib.max_bids(2) + 20
  enc = enc_registry.get(schnaps_enc.INFO_STATE_NAME)
  assert enc.shape == (game_lib.info_state_size(2),) and enc.game == "schnaps"


def test_info_state_matches_template(env):
  """Info state matches template."""
  s = env.init(jax.random.key(0))
  info = env.info_state(s, 1)
  tmpl = schnaps.info_template()
  assert set(info) == set(tmpl)
  for k, v in tmpl.items():
    assert info[k].shape == v.shape and info[k].dtype == v.dtype, k


def test_encoder_equals_the_games_tensor_on_random_states(env):
  """The encoder equals the game's own tensor on every seat of random deals."""
  enc = jax.jit(lambda s, q: schnaps_enc.info_state(env.info_state(s, q)))
  own = jax.jit(env.game.information_state_tensor)
  step = jax.jit(env.step)
  for g in range(8):
    s = env.init(jax.random.key(g))
    rng = np.random.default_rng(g)
    t = 0
    while not bool(env.terminal(s)):
      for q in range(4):
        np.testing.assert_array_equal(
            np.asarray(enc(s, q)), np.asarray(own(s, q))
        )
      legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
      s = step(s, int(rng.choice(legal)), jax.random.key(1000 * g + t))
      t += 1


def test_first_cards_are_the_callers_only(env):
  """Only seat 0 sees which three cards it was dealt first."""
  s = env.init(jax.random.key(4))
  assert int(env.actor(s)) == 0 and int(s.phase) == game_lib.PHASE_CHOOSE_TRUMP
  first = np.asarray(env.info_state(s, 0)["first_cards"])
  assert first.sum() == 3
  np.testing.assert_array_equal(first, np.asarray(env.info_state(s, 0)["hand"]))
  for q in range(1, 4):
    assert not np.asarray(env.info_state(s, q)["first_cards"]).any()


def test_hide_hidden_cards_moves_only_unseen_cards(env):
  """Hide hidden cards keeps every seat's count and the viewer's hand."""
  s = env.init(jax.random.key(3))
  for t in range(10):
    a = int(np.flatnonzero(np.asarray(env.legal_mask(s)))[0])
    s = env.step(s, a, jax.random.key(100 + t))
  h = schnaps.hide_hidden_cards(env, s, 1, jax.random.key(7))
  for q in range(-1, 4):
    assert (np.asarray(h.holder) == q).sum() == (
        np.asarray(s.holder) == q
    ).sum()
  np.testing.assert_array_equal(
      np.asarray(h.holder) == 1, np.asarray(s.holder) == 1
  )
  assert np.asarray(h.first_cards).sum() == 3
  assert not np.array_equal(np.asarray(h.holder), np.asarray(s.holder))


def _lockstep(env, game_str, games, seed):
  """Plays random games of the env and pyspiel in lockstep and compares.

  Args:
    env: The SchnapsEnv.
    game_str: The pyspiel game string.
    games: The games.
    seed: The seed.

  Returns:
    (decision nodes compared, chance nodes compared, terminal returns seen).
  """
  os_game = pyspiel.load_game(game_str)
  apply = jax.jit(env.apply)
  apply_chance = jax.jit(env.apply_chance)
  legal = jax.jit(env.legal_mask)
  actor = jax.jit(env.actor)
  is_chance = jax.jit(env.is_chance)
  probs = jax.jit(env.chance_probs)
  rets = jax.jit(env.returns)
  rng = np.random.default_rng(seed)
  nodes = chance_nodes = 0
  outcomes = set()
  for g in range(games):
    os_s, s = os_game.new_initial_state(), env.root()
    while not os_s.is_terminal():
      ctx = f"{game_str} game {g} history {os_s.history()}"
      assert not bool(env.terminal(s)), ctx
      if os_s.is_chance_node():
        assert bool(is_chance(s)), ctx
        outs, ps = zip(*os_s.chance_outcomes())
        p = np.asarray(probs(s))
        assert set(np.flatnonzero(p).tolist()) == set(outs), ctx
        np.testing.assert_allclose(p[list(outs)], ps, rtol=1e-6, err_msg=ctx)
        a = int(rng.choice(outs, p=ps))
        os_s.apply_action(a)
        s = apply_chance(s, a)
        chance_nodes += 1
        continue
      assert not bool(is_chance(s)), ctx
      assert int(actor(s)) == os_s.current_player(), ctx
      mask = np.asarray(legal(s))
      assert np.flatnonzero(mask).tolist() == sorted(os_s.legal_actions()), ctx
      np.testing.assert_array_equal(
          np.asarray(rets(s)), os_s.returns(), err_msg=ctx
      )
      nodes += 1
      a = int(rng.choice(os_s.legal_actions()))
      os_s.apply_action(a)
      s = apply(s, a)
    assert bool(env.terminal(s)) and int(actor(s)) == -1
    r = np.asarray(rets(s))
    np.testing.assert_array_equal(
        r, os_s.returns(), err_msg=f"{game_str} game {g} returns"
    )
    outcomes.add(tuple(r.tolist()))
  return nodes, chance_nodes, outcomes


@pytest.mark.parametrize("options", OPTION_SETS)
def test_matches_pyspiel_on_random_games(options):
  """Matches OpenSpiel's schnaps on random games."""
  if "schnaps" not in pyspiel.registered_names():
    pytest.skip("the installed pyspiel has no schnaps")
  env = schnaps.SchnapsEnv(**options)
  game_str = "schnaps" + (
      "(" + ",".join(f"{k}={v}" for k, v in options.items()) + ")"
      if options
      else ""
  )
  nodes, chance_nodes, outcomes = _lockstep(
      env, game_str, PYSPIEL_GAMES, seed=7 * len(game_str)
  )
  assert chance_nodes == 20 * PYSPIEL_GAMES
  assert nodes >= PYSPIEL_GAMES * 5
  print(
      f"schnaps vs pyspiel {game_str}: {PYSPIEL_GAMES} games, {nodes} decision"
      f" nodes and {chance_nodes} chance nodes equal, returns seen "
      f"{sorted(outcomes)}"
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider", "-s"]
      )
  )
