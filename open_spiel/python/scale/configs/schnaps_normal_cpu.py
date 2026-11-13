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

r"""Schnaps, Normal game only, self-play sized for a laptop CPU.

configs.schnaps_move_transformer_cpu on the env schnaps_normal (envs.schnaps
SchnapsNormalEnv, OpenSpiel's schnaps(normal_only=true)): pass is the only
bid, so there are no flecks and no called contracts; the trump choice and
the card play are unchanged. Everything else is the CPU config's: the
network, the rollout and learner sizes, the magnet temperature floor and the
in-run evals (vs_uniform; Elena, whose bids fall back to a pass).

  python -m open_spiel.python.scale.train \
      --config=schnaps_normal_cpu \
      --checkpoint_dir=$HOME/schnaps_normal

Running the same command again resumes from the last checkpoint.

Every in-run number is TRACKING tier. Scores are in Normal-game points
(1, 2 or 3 per deal), so they are not comparable with full schnaps runs.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_move_transformer_cpu as cpu
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.envs import schnaps_elena as elena
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.networks import card_move_transformer

NAME = "schnaps_normal_cpu"
ENV = schnaps.SchnapsNormalEnv.name
ELENA_EVAL = elena.NORMAL_EVAL_NAME

NOTES = (
    "schnaps Normal game only (no flecks, no calls) at CPU size: the "
    "schnaps_move_transformer_cpu recipe; in-run evals are TRACKING tier"
)


def policy_kwargs() -> dict[str, object]:
  """The network policy's arch_kwargs: the CPU config's network on ENV.

  Returns:
    The arch_kwargs.
  """
  return common.network_policy_kwargs(
      ENV,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      schnaps.NUM_ACTIONS,
      schnaps.NUM_PLAYERS,
      card_move_transformer.with_options(
          cpu.NETWORK, value_init=full.VALUE_INIT, center_keys=full.CENTER_KEYS
      ),
  )


def evals() -> tuple[runner.InLoopEval, ...]:
  """The CPU config's in-run evals, Elena's under this game's eval name.

  Returns:
    The evals.
  """
  return full.evals(every=cpu.UNIFORM_EVERY, deals=cpu.UNIFORM_DEALS) + (
      runner.InLoopEval(
          ELENA_EVAL,
          every=cpu.ELENA_EVERY,
          deals=cpu.ELENA_DEALS,
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
      cpu.learner(),
      policy_kwargs(),
      num_envs=cpu.NUM_ENVS,
      num_steps=cpu.NUM_STEPS,
      num_updates=cpu.NUM_UPDATES,
      seed=full.SEED,
      keep_every=cpu.KEEP_EVERY,
      env_kwargs={},
      eval=evals(),
      notes=NOTES,
  )
