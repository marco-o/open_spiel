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

"""configs.schnaps_normal_big_gpu: the full-size network with step 3's heads."""

import dataclasses
import sys

import jax
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.configs import schnaps_normal_big_gpu as big
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.configs import schnaps_normal_heads_gpu as heads
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_only_the_network_and_the_start_differ_from_step3():
  cfg = common.load_config(big.NAME)
  assert cfg == big.get_config()
  ref = heads.get_config()
  assert cfg.env == ref.env == "schnaps_normal"
  kw, ref_kw = dict(cfg.policy_kwargs), dict(ref.policy_kwargs)
  net, ref_net = dict(kw.pop("network")), dict(ref_kw.pop("network"))
  assert net.pop("aux_heads") == ref_net.pop("aux_heads")
  assert net == dict(full.policy_kwargs()["network"])
  assert net != ref_net and kw == ref_kw
  assert net["torso"]["depth"] == 6 and net["torso"]["n_head"] == 8
  assert (cfg.league.num_envs, cfg.league.num_steps) == (4096, 84)
  assert cfg.run_updates() == step1.NUM_UPDATES == 3000
  lrn = cfg.learner.learner
  assert lrn.schedule_step_offset == 0
  assert dataclasses.replace(lrn, aux=()) == full.learner(step1.NUM_UPDATES)
  assert lrn.aux == heads.aux_targets()
  assert [(e.name, e.every) for e in cfg.evals()] == [
      (e.name, e.every) for e in step1.get_config().evals()
  ]


def test_the_network_builds_with_its_heads():
  x64.enable_x64()
  _, params = refs.PolicyRef(
      arch_name="network", arch_kwargs=big.policy_kwargs()
  ).build(jax.random.key(0))
  keys = {"/".join(str(getattr(p, "key", p)) for p in path)
          for path, _ in jax.tree_util.tree_leaves_with_path(params)}
  assert any("aux_holder" in k for k in keys)
  assert any("aux_score" in k for k in keys)
  n = sum(x.size for x in jax.tree_util.tree_leaves(params))
  assert n > 3_000_000, n


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
