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

"""configs.schnaps_move_transformer_cpu: the CPU sizes of the schnaps recipe."""

import dataclasses
import sys

import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_cpu as cpu,
)
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.league import config_common as common

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_cpu_sizes_of_the_full_recipe():
  cfg = common.load_config(cpu.NAME)
  assert cfg == cpu.get_config()
  ref = full.get_config()
  assert cfg.env == ref.env and dict(cfg.env_kwargs) == dict(ref.env_kwargs)
  assert cfg.league.self_play_seats == "all"
  assert (cfg.league.num_envs, cfg.league.num_steps) == (128, 84)
  assert cfg.run_updates() == 2000 and cfg.keep_every == 25
  lc, rc = cfg.learner.learner, ref.learner.learner
  assert (lc.loss, lc.credit, lc.value_scale, lc.value_rows) == (
      rc.loss,
      rc.credit,
      rc.value_scale,
      rc.value_rows,
  )
  assert (lc.minibatch_size, lc.sgd_steps) == (2048, 8)
  assert lc.loss_config.temperature_floor == 0.015
  assert lc.loss_config.field_players == 0
  # The rest of the loss is the recipe's.
  assert dataclasses.replace(
      lc.loss_config, temperature_floor=rc.loss_config.temperature_floor
  ) == rc.loss_config
  assert [(e.name, e.every) for e in cfg.evals()] == [
      ("vs_uniform", 50),
      ("schnaps.elena", 50),
  ]
  net = cfg.policy_kwargs["network"]
  torso = net["torso"]
  assert (torso["depth"], torso["n_head"], torso["in_channels"]) == (2, 4, 70)
  assert (torso["n_card"], torso["n_special_action"]) == (20, 19)
  assert net["compute_dtype"] == "float32" and torso["center_keys"]
  assert net["value_head"]["init_scheme"] == "torch_uniform"


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
