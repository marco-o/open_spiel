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

r"""Schnaps, Normal game only, at the full recipe's batch on one GPU.

Step 1 of Docker/gpu-training.md: the network of configs.schnaps_normal_cpu
(the compact CPU card move transformer, unchanged) trained at the batch the
recipe was tuned for, to measure what the batch alone buys.

Compared with schnaps_normal_cpu:
  * rollout: 4,096 tables x 84 plies per update (the CPU config: 128).
  * learner: configs.schnaps_move_transformer_self_play's (128 SGD steps of
    at most 4,096 plies; the magnet temperature's default floor 0.001). The
    CPU config's raised floor (0.015) was a fix for its small batch.
  * length: 3,000 updates, a keep every 100.
  * in-run evals: vs_uniform (512 deals) and Elena (256 deal pairs), both
    every 100 updates.
Everything else (env, network, seed, value rows, value scale) is the same.

  python -m open_spiel.python.scale.train \
      --config=schnaps_normal_gpu \
      --checkpoint_dir=/workspace/schnaps_normal_gpu

A benchmark of a few updates: add --num_updates=20. The log reports
update_seconds and frames_per_s for each update.

Every in-run number is TRACKING tier. Scores are in Normal-game points
(1, 2 or 3 per deal), comparable with schnaps_normal_cpu.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_move_transformer_cpu as cpu
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.league import runner

NAME = "schnaps_normal_gpu"
ENV = normal.ENV
NUM_ENVS = full.NUM_ENVS
NUM_STEPS = full.NUM_STEPS
NUM_UPDATES = 3000
KEEP_EVERY = 100
EVAL_EVERY = 100
UNIFORM_DEALS = full.UNIFORM_DEALS
ELENA_DEALS = cpu.ELENA_DEALS

NOTES = (
    "schnaps Normal game only at the full recipe's batch (4,096 tables) on "
    "one GPU, with the CPU config's network; in-run evals are TRACKING tier"
)


def evals() -> tuple[runner.InLoopEval, ...]:
  """The in-run evals: vs_uniform and Elena, every EVAL_EVERY updates.

  Returns:
    The evals.
  """
  return full.evals(every=EVAL_EVERY, deals=UNIFORM_DEALS) + (
      runner.InLoopEval(
          normal.ELENA_EVAL,
          every=EVAL_EVERY,
          deals=ELENA_DEALS,
          seed=cpu.ELENA_SEED,
      ),
  )


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.ppo_magnet_self_play(
      NAME,
      ENV,
      full.learner(NUM_UPDATES),
      normal.policy_kwargs(),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=NUM_UPDATES,
      seed=full.SEED,
      keep_every=KEEP_EVERY,
      env_kwargs={},
      eval=evals(),
      notes=NOTES,
  )
