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

"""configs.schnaps_forced_*_cpu and the warm start's relabelling."""

import dataclasses
import sys

import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import schnaps_forced_cpu as forced
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.examples import schnaps_relabel_checkpoint as relabel
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs

# pylint: enable=g-import-not-at-top,wrong-import-position


@pytest.mark.parametrize("contract", forced.CONTRACTS)
def test_the_normal_recipe_on_the_forced_env(contract):
  cfg = common.load_config(f"schnaps_forced_{contract}_cpu")
  assert cfg == forced.config(contract)
  ref = normal.get_config()
  assert cfg.env == f"schnaps_forced_{contract}" and dict(cfg.env_kwargs) == {}
  assert (cfg.league.num_envs, cfg.league.num_steps) == (128, 84)
  assert cfg.run_updates() == forced.NUM_UPDATES == 1000
  assert cfg.learner.init == refs.PolicyRef(
      artifact=f"{forced.INIT_DIR}/{contract}@{forced.INIT_STEP}"
  )
  learner, ref_learner = cfg.learner.learner, ref.learner.learner
  assert learner.schedule_step_offset == forced.INIT_STEP
  assert dataclasses.replace(
      learner, schedule_step_offset=0, num_updates=ref_learner.num_updates
  ) == ref_learner
  assert [(e.name, e.every) for e in cfg.evals()] == [
      ("vs_uniform", 50),
      (f"schnaps_forced_{contract}.elena", 50),
  ]
  kw, ref_kw = dict(cfg.policy_kwargs), dict(ref.policy_kwargs)
  assert kw.pop("game") == cfg.env
  ref_kw.pop("game")
  assert kw == ref_kw


def test_relabel_makes_a_warm_start_of_the_learners_spec(tmp_path):
  x64.enable_x64()
  env = env_registry.make("schnaps_normal")
  policy, params = refs.PolicyRef(
      arch_name="network", arch_kwargs=normal.policy_kwargs()
  ).build(jax.random.key(1))
  src = str(tmp_path / "normal")
  artifact.write_artifact(
      src,
      4000,
      policy,
      params,
      rules_sha=env.declared_rules_sha,
      env_spec=env_registry.spec_of(env).to_json(),
      env_name=env.name,
      extensions={"value_scale": 3.0},
  )
  dst = f"{tmp_path}/init@4000"
  assert relabel.relabel("lomt", src, 4000, dst) == dst
  art = artifact.load_artifact(str(tmp_path / "init"), 4000)
  assert art.manifest.game == "schnaps_forced_lomt"
  assert art.manifest.parents == (f"{artifact.norm_path(src)}@4000",)
  assert art.manifest.extensions["value_scale"] == 3.0
  for a, b in zip(
      jax.tree_util.tree_leaves(art.params), jax.tree_util.tree_leaves(params)
  ):
    assert np.array_equal(np.asarray(a), np.asarray(b))
  learner_policy, _ = refs.PolicyRef(
      arch_name="network", arch_kwargs=forced.policy_kwargs("lomt")
  ).build()
  assert art.policy.spec == learner_policy.spec
  # A second run leaves the written step alone.
  assert relabel.relabel("lomt", src, 4000, dst) == dst


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
