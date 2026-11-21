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

r"""Schnaps, one full-size network for every contract, on one GPU.

Step 2a of Docker/gpu-training.md: the step-4 network trained on the mixed
env (envs.schnaps_mixed: half Normal deals with the Caller's trump choice, a
sixth each of forced Schnapser, Lomt and Bauern deals), so one network plays
what the Normal network and the three forced-contract networks play today.
The bidder (envs.schnaps_bidding) is unchanged; it reads this one network's
value head for every contract.

  * network: configs.schnaps_normal_big_gpu's (depth 6, 8 heads, D = 256,
    bf16 matmuls), warm-started from ~/schnaps_normal_big_gpu@1000.
  * aux heads: the card-holder head (kept from the source) and the
    final-score head, widened from +-3 to +-SCORE_MAX points (a Bauern is
    worth 12), so its params start fresh.
  * learner: the full recipe's (as step 4), the schedules continued from
    update INIT_STEP, plus the aux targets.
  * rollout and keeps: step 1's (4,096 tables x 84 plies, a keep every 100).
  * in-run evals: vs_uniform and Elena on the mixed deals, every 100.

The warm start needs the learner's spec, so
examples/schnaps_mixed_init.py writes the source's params under this
config's spec (the mixed env, the widened score head) to init_artifact():

  python -m open_spiel.python.scale.examples.schnaps_mixed_init
  python -m open_spiel.python.scale.train \
      --config=schnaps_mixed_big_gpu \
      --checkpoint_dir=/workspace/schnaps_mixed_big_gpu

Measured on the laptop against the step-4 network and the forced-contract
networks (Docker/gpu-training.md, "What 2a must show"). Every in-run number
is TRACKING tier.
"""

import dataclasses
import os

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_normal_big_gpu as big
from open_spiel.python.scale.configs import schnaps_move_transformer_cpu as cpu
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.configs import schnaps_normal_heads_gpu as heads
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import card_move_transformer
from open_spiel.python.scale.networks import catalog

NAME = "schnaps_mixed_big_gpu"
ENV = "schnaps_mixed"
NETWORK = big.NETWORK
NUM_UPDATES = 1000
KEEP_EVERY = step1.KEEP_EVERY
EVAL_EVERY = step1.EVAL_EVERY
#: The mixed env's Elena eval (envs.schnaps_elena MIXED_EVAL_NAME).
ELENA_EVAL = "schnaps_mixed.elena"
#: The step-4 network the run starts from.
SRC_DIR = os.path.expanduser("~/schnaps_normal_big_gpu")
INIT_STEP = 1000
INIT_DIR = os.path.expanduser("~/schnaps_mixed_init")
#: The final-score head's range: a Bauern, the biggest contract here, is 12.
SCORE_MAX = 12

NOTES = (
    "schnaps, one full-size network for the Normal game and forced "
    "Schnapser, Lomt and Bauern deals (the mixed env), warm-started from the "
    "step-4 network; in-run evals are TRACKING tier"
)


def init_artifact() -> str:
  """The warm start's artifact id: the step-4 network under this spec."""
  return f"{INIT_DIR}@{INIT_STEP}"


def aux_heads() -> list[dict[str, object]]:
  """Step 3's aux heads with the score head widened to +-SCORE_MAX."""
  return [
      dict(key=heads.HOLDER, source="cards", features=4, squeeze=False),
      dict(
          key=heads.SCORE,
          source="value",
          features=2 * SCORE_MAX + 1,
          squeeze=False,
      ),
  ]


def aux_targets() -> tuple[pg.AuxTarget, ...]:
  """Step 3's aux targets with the score range widened."""
  return (
      pg.AuxTarget(
          heads.HOLDER,
          "token_class",
          coef=heads.HOLDER_COEF,
          label=heads.HOLDER_LABEL,
      ),
      pg.AuxTarget(
          heads.SCORE, "final_score", coef=heads.SCORE_COEF, score_min=-SCORE_MAX
      ),
  )


def policy_kwargs() -> dict[str, object]:
  """The step-4 network on ENV, with the widened aux heads.

  Returns:
    The arch_kwargs.
  """
  network = card_move_transformer.with_options(
      catalog.get(NETWORK),
      value_init=full.VALUE_INIT,
      center_keys=full.CENTER_KEYS,
  )
  assert "aux_heads" not in network, network
  network["aux_heads"] = aux_heads()
  return common.network_policy_kwargs(
      ENV,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      schnaps.NUM_ACTIONS,
      schnaps.NUM_PLAYERS,
      network,
  )


def learner(num_updates: int = NUM_UPDATES) -> pg.PGConfig:
  """The full recipe's learner, continued from INIT_STEP, plus the heads.

  Args:
    num_updates: the run's updates.

  Returns:
    The pg.PGConfig.
  """
  return dataclasses.replace(
      full.learner(num_updates),
      schedule_step_offset=INIT_STEP,
      aux=aux_targets(),
  )


def evals() -> tuple[runner.InLoopEval, ...]:
  """vs_uniform and Elena on the mixed deals, every EVAL_EVERY updates."""
  return full.evals(every=EVAL_EVERY, deals=step1.UNIFORM_DEALS) + (
      runner.InLoopEval(
          ELENA_EVAL,
          every=EVAL_EVERY,
          deals=step1.ELENA_DEALS,
          seed=cpu.ELENA_SEED,
      ),
  )


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  init = init_artifact()
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
      init=refs.PolicyRef(artifact=init),
      parents=(init,),
      eval=evals(),
      notes=NOTES,
  )
