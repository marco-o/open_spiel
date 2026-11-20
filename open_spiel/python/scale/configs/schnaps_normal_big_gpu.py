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

r"""Schnaps, Normal game only, with the full-size network, on one GPU.

Step 4 of Docker/gpu-training.md: configs.schnaps_normal_heads_gpu with the
compact CPU network replaced by the full recipe's, trained from scratch (the
layer sizes change, so no warm start):

  * network: catalog NETWORK (configs.schnaps_move_transformer_self_play's:
    depth 6, 8 heads, D = 256, FFN 4D, bf16 matmuls, f32 params; the CPU
    network is depth 2, 4 heads, D = 128, FFN 2D, f32), with the recipe's
    value init and centred keys.
  * aux heads: step 3's card-holder and final-score heads (learners.pg
    "Aux heads"), same loss weights.
  * learner: the full recipe's (as step 1), plus the aux targets; the
    schedules start at 0.
  * rollout, length, keeps and in-run evals: step 1's (4,096 tables x 84
    plies, 3,000 updates, a keep and both evals every 100).

The card planes are already seat-relative (encoders.schnaps_card_planes),
so the plan's "seats relative to the player" needs nothing here. The
opponent pool is left for a later run, so this one changes only the network.

  python -m open_spiel.python.scale.train \
      --config=schnaps_normal_big_gpu \
      --checkpoint_dir=/workspace/schnaps_normal_big_gpu

A benchmark of a few updates: add --num_updates=10. Measured against step 1
at 2700 and step 3 (+0.39 and +0.38 +- 0.02 per deal against Elena, 2,048
deal pairs). Every in-run number is TRACKING tier.
"""

import dataclasses

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.configs import schnaps_normal_heads_gpu as heads
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import card_move_transformer
from open_spiel.python.scale.networks import catalog

NAME = "schnaps_normal_big_gpu"
ENV = step1.ENV
NETWORK = full.NETWORK
NUM_UPDATES = step1.NUM_UPDATES
KEEP_EVERY = step1.KEEP_EVERY

NOTES = (
    "schnaps Normal game only, the full-size card move transformer from "
    "scratch with the card-holder and final-score heads, 4,096 tables on one "
    "GPU; in-run evals are TRACKING tier"
)


def policy_kwargs() -> dict[str, object]:
  """The full-size network on ENV, with step 3's aux heads.

  Returns:
    The arch_kwargs.
  """
  network = card_move_transformer.with_options(
      catalog.get(NETWORK),
      value_init=full.VALUE_INIT,
      center_keys=full.CENTER_KEYS,
  )
  assert "aux_heads" not in network, network
  network["aux_heads"] = heads.aux_heads()
  return common.network_policy_kwargs(
      ENV,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      schnaps.NUM_ACTIONS,
      schnaps.NUM_PLAYERS,
      network,
  )


def learner(num_updates: int = NUM_UPDATES) -> pg.PGConfig:
  """The full recipe's learner plus step 3's aux targets.

  Args:
    num_updates: the run's updates.

  Returns:
    The pg.PGConfig.
  """
  return dataclasses.replace(full.learner(num_updates), aux=heads.aux_targets())


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.ppo_magnet_self_play(
      NAME,
      ENV,
      learner(),
      policy_kwargs(),
      num_envs=step1.NUM_ENVS,
      num_steps=step1.NUM_STEPS,
      num_updates=NUM_UPDATES,
      seed=full.SEED,
      keep_every=KEEP_EVERY,
      env_kwargs={},
      eval=step1.evals(),
      notes=NOTES,
  )
