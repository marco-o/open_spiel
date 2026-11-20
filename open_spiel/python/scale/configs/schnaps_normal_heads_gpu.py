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

r"""Schnaps, Normal game only, with two extra output heads, on one GPU.

Step 3 (heads 1 and 2) of Docker/gpu-training.md: configs.schnaps_normal_gpu
continued from its network at INIT_STEP with two aux heads the learner trains
next to the loss (learners.pg "Aux heads"):

  * holder: for each of the 20 cards, which seat holds it, relative to the
    player (env label holder_rel: 0 itself, 2 the partner, 1 and 3 the
    opponents). One 4-way softmax per card token. It costs almost nothing and
    pushes the network to read the declarations and the cards not played.
  * score: the deal's final score from the player's side, a softmax over
    -SCORE_MAX .. +SCORE_MAX points (the Normal game's are +-1, +-2, +-3).
    It is trained on the score the deal really ended with, so it is the
    score's distribution (win probability, capped values), not a TD
    estimate. The value head is unchanged.

The heads are new params; examples/schnaps_add_heads.py writes the step-1
network with them freshly initialised to INIT_DIR@INIT_STEP:

  python -m open_spiel.python.scale.examples.schnaps_add_heads
  python -m open_spiel.python.scale.train \
      --config=schnaps_normal_heads_gpu \
      --checkpoint_dir=/workspace/schnaps_normal_heads_gpu

The learner's schedules continue from update INIT_STEP
(schedule_step_offset). Everything else (env, network body, batch, learner,
evals) is schnaps_normal_gpu's, so the run compares with that network
(+0.39 +- 0.02 per deal against Elena, 2,048 deal pairs, at step 2700).

Every in-run number is TRACKING tier.
"""

import dataclasses
import os

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.configs import schnaps_normal_gpu as step1
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg

NAME = "schnaps_normal_heads_gpu"
ENV = step1.ENV
NUM_UPDATES = 1000
KEEP_EVERY = 100
#: The step-1 run and its network the heads are added to.
STEP1_DIR = os.path.expanduser("~/schnaps_normal_gpu")
INIT_STEP = 2700
INIT_DIR = os.path.expanduser("~/schnaps_normal_heads_init")
HOLDER = "holder"
HOLDER_LABEL = "holder_rel"
SCORE = "score"
SCORE_MAX = 3
HOLDER_COEF = 0.5
SCORE_COEF = 0.5

NOTES = (
    "schnaps Normal game only, the step-1 GPU network continued with a "
    "card-holder head and a final-score head; in-run evals are TRACKING tier"
)


def init_artifact() -> str:
  """The warm start's artifact id: the step-1 network with the heads added."""
  return f"{INIT_DIR}@{INIT_STEP}"


def aux_heads() -> list[dict[str, object]]:
  """The network's aux heads (actor_critic "aux_heads")."""
  return [
      dict(key=HOLDER, source="cards", features=4, squeeze=False),
      dict(key=SCORE, source="value", features=2 * SCORE_MAX + 1, squeeze=False),
  ]


def policy_kwargs() -> dict[str, object]:
  """schnaps_normal_cpu's arch_kwargs with the aux heads added.

  Returns:
    The arch_kwargs.
  """
  kw = normal.policy_kwargs()
  network = dict(kw["network"])
  assert "aux_heads" not in network, network
  network["aux_heads"] = aux_heads()
  return dict(kw, network=network)


def aux_targets() -> tuple[pg.AuxTarget, ...]:
  """What the learner trains the heads on (learners.pg "Aux heads")."""
  assert planes.N_CARD == 20
  return (
      pg.AuxTarget(
          HOLDER, "token_class", coef=HOLDER_COEF, label=HOLDER_LABEL
      ),
      pg.AuxTarget(SCORE, "final_score", coef=SCORE_COEF, score_min=-SCORE_MAX),
  )


def learner(num_updates: int = NUM_UPDATES) -> pg.PGConfig:
  """schnaps_normal_gpu's learner, continued from INIT_STEP, plus the heads.

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
      eval=step1.evals(),
      notes=NOTES,
  )
