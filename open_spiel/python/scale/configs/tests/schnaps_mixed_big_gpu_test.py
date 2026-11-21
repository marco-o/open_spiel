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

"""configs.schnaps_mixed_big_gpu and its warm start (examples.schnaps_mixed_init)."""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.configs import schnaps_mixed_big_gpu as mixed
from open_spiel.python.scale.configs import schnaps_normal_big_gpu as big
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.examples import schnaps_mixed_init as init_lib
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs

# pylint: enable=g-import-not-at-top,wrong-import-position


@pytest.fixture(scope="module", autouse=True)
def _x64():
  x64.enable_x64()


def test_only_the_env_the_score_head_and_the_start_differ_from_step4():
  cfg = common.load_config(mixed.NAME)
  assert cfg == mixed.get_config()
  ref = big.get_config()
  assert cfg.env == "schnaps_mixed" and ref.env == "schnaps_normal"
  kw, ref_kw = dict(cfg.policy_kwargs), dict(ref.policy_kwargs)
  assert kw.pop("game") == "schnaps_mixed" and ref_kw.pop("game")
  net, ref_net = dict(kw.pop("network")), dict(ref_kw.pop("network"))
  heads, ref_heads = net.pop("aux_heads"), ref_net.pop("aux_heads")
  assert net == ref_net and kw == ref_kw
  assert heads[0] == ref_heads[0]
  assert heads[1] == dict(ref_heads[1], features=25)
  assert (cfg.league.num_envs, cfg.league.num_steps) == (4096, 84)
  assert cfg.run_updates() == mixed.NUM_UPDATES == 1000
  lrn = cfg.learner.learner
  assert lrn.schedule_step_offset == mixed.INIT_STEP == 1000
  assert dataclasses.replace(lrn, aux=(), schedule_step_offset=0) == (
      full.learner(mixed.NUM_UPDATES)
  )
  assert lrn.aux[0] == big.learner().aux[0]
  assert lrn.aux[1].score_min == -12
  assert [e.name for e in cfg.evals()] == [
      e.name for e in step1.evals()[:-1]
  ] + ["schnaps_mixed.elena"]
  assert cfg.learner.init == refs.PolicyRef(artifact=mixed.init_artifact())


def _write_source(tmp_path):
  """A random step-4 network written as an artifact; returns its dir."""
  kwargs = big.policy_kwargs()
  policy, params = refs.PolicyRef(
      arch_name="network", arch_kwargs=kwargs
  ).build(jax.random.key(3))
  env = env_registry.make(big.ENV)
  src = tmp_path / "big"
  artifact.write_artifact(
      str(src),
      7,
      policy,
      params,
      rules_sha=env.declared_rules_sha,
      env_spec=env_registry.spec_of(env).to_json(),
      default_rule=types.ARGMAX,
      env_name=env.name,
      extensions={"value_scale": 3.0},
  )
  return str(src), params


def test_the_warm_start_keeps_everything_but_the_score_head(tmp_path):
  src_dir, src_params = _write_source(tmp_path)
  dst = f"{tmp_path / 'init'}@7"
  init_lib.init(src_dir, 7, dst)
  dst_dir, _ = dst.rsplit("@", 1)
  art = artifact.load_artifact(dst_dir, 7)
  assert art.manifest.game == "schnaps_mixed"
  assert dict(art.manifest.arch_kwargs) == mixed.policy_kwargs()
  flat = lambda t: {  # noqa: E731
      init_lib._key(p): x  # pylint: disable=protected-access
      for p, x in jax.tree_util.tree_flatten_with_path(t)[0]
  }
  got, src = flat(art.params), flat(src_params)
  fresh = [k for k in got if init_lib._fresh(k)]  # pylint: disable=protected-access
  assert fresh and any(got[k].shape[-1:] == (25,) for k in fresh)
  for k, v in src.items():
    if not init_lib._fresh(k):  # pylint: disable=protected-access
      assert np.array_equal(np.asarray(v), np.asarray(got[k])), k
  # A second call leaves the target alone.
  assert init_lib.init(src_dir, 7, dst) == dst


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
