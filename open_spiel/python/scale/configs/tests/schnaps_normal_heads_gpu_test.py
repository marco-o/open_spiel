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

"""configs.schnaps_normal_heads_gpu: step 1's network plus two aux heads."""

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
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.configs import schnaps_normal_heads_gpu as heads
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.examples import schnaps_add_heads
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_only_the_heads_and_the_warm_start_differ_from_step1():
  cfg = common.load_config(heads.NAME)
  assert cfg == heads.get_config()
  ref = step1.get_config()
  assert cfg.env == ref.env == "schnaps_normal"
  kw, ref_kw = dict(cfg.policy_kwargs), dict(ref.policy_kwargs)
  net = dict(kw.pop("network"))
  assert [h["key"] for h in net.pop("aux_heads")] == ["holder", "score"]
  assert net == dict(ref_kw.pop("network")) and kw == ref_kw
  assert (cfg.league.num_envs, cfg.league.num_steps) == (4096, 84)
  assert cfg.run_updates() == heads.NUM_UPDATES
  lrn = cfg.learner.learner
  assert dataclasses.replace(lrn, aux=(), schedule_step_offset=0) == (
      full.learner(heads.NUM_UPDATES)
  )
  assert lrn.schedule_step_offset == heads.INIT_STEP
  assert [(t.name, t.kind) for t in lrn.aux] == [
      ("holder", "token_class"),
      ("score", "final_score"),
  ]
  assert [(e.name, e.every) for e in cfg.evals()] == [
      (e.name, e.every) for e in ref.evals()
  ]


def test_the_heads_fit_the_network_and_the_merge_keeps_the_body():
  x64.enable_x64()
  _, fresh = refs.PolicyRef(
      arch_name="network", arch_kwargs=heads.policy_kwargs()
  ).build(jax.random.key(0))
  _, body = refs.PolicyRef(
      arch_name="network", arch_kwargs=step1.get_config().policy_kwargs
  ).build(jax.random.key(1))
  merged = schnaps_add_heads.merge(fresh, body)
  f, b, m = (schnaps_add_heads._flat(t) for t in (fresh, body, merged))  # pylint: disable=protected-access
  added = set(f) - set(b)
  assert added and all("aux_" in k for k in added)
  for k in b:
    assert (m[k] == b[k]).all()
  for k in added:
    assert (m[k] == f[k]).all()


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
