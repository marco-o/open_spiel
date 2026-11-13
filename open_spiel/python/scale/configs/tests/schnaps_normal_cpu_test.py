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

"""configs.schnaps_normal_cpu: the CPU config on the Normal-only env."""

import sys

import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import schnaps_move_transformer_cpu as cpu
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_the_cpu_config_on_the_normal_env():
  cfg = common.load_config(normal.NAME)
  assert cfg == normal.get_config()
  ref = cpu.get_config()
  assert cfg.env == "schnaps_normal" and dict(cfg.env_kwargs) == {}
  assert cfg.learner == ref.learner
  assert (cfg.league.num_envs, cfg.league.num_steps) == (128, 84)
  assert cfg.run_updates() == ref.run_updates()
  assert [(e.name, e.every) for e in cfg.evals()] == [
      ("vs_uniform", 50),
      ("schnaps_normal.elena", 50),
  ]
  assert cfg.policy_kwargs["network"] == ref.policy_kwargs["network"]


def test_the_env_allows_only_passes_in_the_bidding():
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  env = env_registry.make("schnaps_normal")
  g = env.game
  rng = jax.random.key(0)
  for deal in range(20):
    s = g.new_initial_state()
    k = jax.random.fold_in(rng, deal)
    n = 0
    while not bool(g.is_terminal(s)):
      n += 1
      k, sub = jax.random.split(k)
      if bool(g.is_chance_node(s)):
        probs = g.chance_outcomes(s)
        a = jax.random.choice(sub, probs.shape[0], p=probs)
      else:
        legal = g.legal_actions_mask(s)
        if int(s.phase) == 3:  # bidding
          assert [int(i) for i in jnp.flatnonzero(legal)] == [32]
        a = jax.random.choice(sub, legal.shape[0], p=legal / legal.sum())
      s = g.apply_action(s, jnp.int32(a))
    assert int(s.contract) == 0  # the Normal game
    assert abs(float(g.returns(s)[0])) in (1.0, 2.0, 3.0)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
