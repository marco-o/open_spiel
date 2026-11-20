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

"""configs.schnaps_normal_gpu: the Normal CPU network at the full batch."""

import sys

import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.configs import schnaps_normal_gpu as gpu
from open_spiel.python.scale.league import config_common as common

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_only_the_batch_and_the_learner_differ_from_the_cpu_config():
  cfg = common.load_config(gpu.NAME)
  assert cfg == gpu.get_config()
  ref = normal.get_config()
  assert cfg.env == ref.env == "schnaps_normal"
  assert dict(cfg.env_kwargs) == {}
  assert dict(cfg.policy_kwargs) == dict(ref.policy_kwargs)
  assert (cfg.league.num_envs, cfg.league.num_steps) == (4096, 84)
  assert cfg.run_updates() == gpu.NUM_UPDATES == 3000
  assert cfg.learner.learner == full.learner(gpu.NUM_UPDATES)
  assert cfg.learner.learner.sgd_steps == full.SGD_STEPS
  assert [(e.name, e.every) for e in cfg.evals()] == [
      ("vs_uniform", 100),
      ("schnaps_normal.elena", 100),
  ]


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
