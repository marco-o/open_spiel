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

r"""Forced-contract card-play runs at CPU size: the shared recipe.

Not a config itself (no get_config): schnaps_forced_<contract>_cpu are, one
per contract (Schnapser, Lomt, Bauern). Each is configs.schnaps_normal_cpu's
recipe on the env schnaps_forced_<contract> (envs.schnaps_forced), except:

  * warm start: the Normal-only network at step INIT_STEP, re-saved for the
    forced env by examples/schnaps_relabel_checkpoint.py into
    INIT_DIR/<contract>@INIT_STEP (the network spec names its env, so the
    Normal checkpoint cannot be the init as it is);
  * the learner's schedules continue from update INIT_STEP
    (schedule_step_offset), so the step size and the magnet temperature carry
    on from where the Normal run stopped instead of restarting;
  * NUM_UPDATES (1,000) updates;
  * evals: vs_uniform and Elena on the forced deals
    (schnaps_forced_<contract>.elena), every 50 updates.

The value scale stays 3: the value head is linear, so Lomt's ±9 and Bauern's
±12 are targets of ±3 and ±4, in the units the warm-started head knows.

  python -m open_spiel.python.scale.examples.schnaps_relabel_checkpoint \
      --contract=lomt
  python -m open_spiel.python.scale.train \
      --config=schnaps_forced_lomt_cpu \
      --checkpoint_dir=$HOME/schnaps_forced_lomt

Every in-run number is TRACKING tier.
"""

import os

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import schnaps_move_transformer_cpu as cpu
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import ppo_magnet
from open_spiel.python.scale.networks import card_move_transformer

CONTRACTS = ("schnapser", "lomt", "bauern")
#: The Normal-only run's step the fine-tunes start from.
INIT_STEP = 4000
NORMAL_DIR = os.path.expanduser("~/schnaps_normal")
INIT_DIR = os.path.expanduser("~/schnaps_forced_init")
NUM_UPDATES = 1000


def env_name(contract: str) -> str:
  """The forced env's name (envs.schnaps_forced)."""
  if contract not in CONTRACTS:
    raise ValueError(f"unknown contract {contract!r}; known: {CONTRACTS}")
  return f"schnaps_forced_{contract}"


def config_name(contract: str) -> str:
  return f"{env_name(contract)}_cpu"


def elena_eval(contract: str) -> str:
  """The forced env's Elena eval (envs.schnaps_elena FORCED_EVAL_NAMES)."""
  return f"{env_name(contract)}.elena"


def init_artifact(contract: str) -> str:
  """The warm start's artifact id: the relabelled Normal network."""
  env_name(contract)
  return f"{INIT_DIR}/{contract}@{INIT_STEP}"


def policy_kwargs(contract: str) -> dict[str, object]:
  """The CPU network's arch_kwargs on the forced env.

  Args:
    contract: the contract.

  Returns:
    The arch_kwargs (configs.schnaps_normal_cpu's but for `game`).
  """
  return common.network_policy_kwargs(
      env_name(contract),
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      schnaps.NUM_ACTIONS,
      schnaps.NUM_PLAYERS,
      card_move_transformer.with_options(
          cpu.NETWORK, value_init=full.VALUE_INIT, center_keys=full.CENTER_KEYS
      ),
  )


def learner(offset: int = INIT_STEP, num_updates: int = NUM_UPDATES):
  """The CPU learner with its schedules continued from update `offset`.

  Args:
    offset: the update the schedules continue from.
    num_updates: the run's updates.

  Returns:
    The pg.PGConfig.
  """
  return recipes.ppo_magnet_config(
      num_updates=num_updates,
      sgd_steps=cpu.SGD_STEPS,
      value_scale=full.VALUE_SCALE,
      minibatch_size=cpu.MINIBATCH,
      forward_chunk_rows=cpu.FORWARD_CHUNK,
      value_rows=full.VALUE_ROWS,
      max_decision_share=full.MAX_DECISION_SHARE,
      schedule_step_offset=offset,
      loss_config=ppo_magnet.PPOMagnetConfig(
          temperature_floor=cpu.TEMPERATURE_FLOOR
      ),
  )


def evals(contract: str) -> tuple[runner.InLoopEval, ...]:
  """vs_uniform and Elena on the forced deals.

  Args:
    contract: the contract.

  Returns:
    The evals.
  """
  return full.evals(every=cpu.UNIFORM_EVERY, deals=cpu.UNIFORM_DEALS) + (
      runner.InLoopEval(
          elena_eval(contract),
          every=cpu.ELENA_EVERY,
          deals=cpu.ELENA_DEALS,
          seed=cpu.ELENA_SEED,
      ),
  )


def config(contract: str) -> runner.LeagueRunConfig:
  """The run config of one contract (module doc).

  Args:
    contract: "schnapser", "lomt" or "bauern".

  Returns:
    The runner.LeagueRunConfig.
  """
  init = init_artifact(contract)
  return recipes.ppo_magnet_self_play(
      config_name(contract),
      env_name(contract),
      learner(),
      policy_kwargs(contract),
      num_envs=cpu.NUM_ENVS,
      num_steps=cpu.NUM_STEPS,
      num_updates=NUM_UPDATES,
      seed=full.SEED,
      keep_every=cpu.KEEP_EVERY,
      env_kwargs={},
      init=refs.PolicyRef(artifact=init),
      parents=(init,),
      eval=evals(contract),
      notes=(
          f"schnaps card play with {contract} forced, fine-tuned from the "
          f"Normal-only network at step {INIT_STEP}; in-run evals are "
          "TRACKING tier"
      ),
  )


#: Round 2 (Schnapser after its rule change): fine-tune a forced network
#: further, from its own round-1 checkpoint.
ROUND2_UPDATES = 500


def round1_artifact(contract: str) -> str:
  """The round-1 network of `contract`: ~/schnaps_forced_<contract>@1000."""
  env_name(contract)
  return f"{os.path.expanduser('~')}/schnaps_forced_{contract}@{NUM_UPDATES}"


def config_round2(contract: str) -> runner.LeagueRunConfig:
  """A second fine-tune of `contract` from its round-1 network.

  Made for Schnapser after its rule became "win each of the first three
  tricks" (games/schnaps): the round-1 network learned the old rule (66
  within three tricks), so its play and value head are off. The run warm
  starts from round1_artifact (the same env and spec), continues the
  schedules from update INIT_STEP + NUM_UPDATES, and trains ROUND2_UPDATES
  updates on the current hand weights (envs/schnaps_forced_hands.json).

  Args:
    contract: the contract.

  Returns:
    The runner.LeagueRunConfig.
  """
  init = round1_artifact(contract)
  return recipes.ppo_magnet_self_play(
      f"{config_name(contract)[: -len('_cpu')]}_r2_cpu",
      env_name(contract),
      learner(INIT_STEP + NUM_UPDATES, ROUND2_UPDATES),
      policy_kwargs(contract),
      num_envs=cpu.NUM_ENVS,
      num_steps=cpu.NUM_STEPS,
      num_updates=ROUND2_UPDATES,
      seed=full.SEED + 1,
      keep_every=cpu.KEEP_EVERY,
      env_kwargs={},
      init=refs.PolicyRef(artifact=init),
      parents=(init,),
      eval=evals(contract),
      notes=(
          f"schnaps card play with {contract} forced, round 2: fine-tuned "
          f"from its round-1 network ({init}) under the current rules; "
          "in-run evals are TRACKING tier"
      ),
  )
