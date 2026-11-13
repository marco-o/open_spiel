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

r"""Schnaps self-play sized for a laptop CPU.

The recipe of configs.schnaps_move_transformer_self_play (one card move
transformer on all four seats, ppo_magnet, team reward, waiting value rows)
at a size a few CPU cores train in days rather than months: a compact f32
network, NUM_ENVS tables and small SGD steps.

  python -m open_spiel.python.scale.train \
      --config=schnaps_move_transformer_cpu \
      --checkpoint_dir=$HOME/schnaps_cpu

Running the same command again resumes from the last checkpoint (one every
KEEP_EVERY updates), so the run can be stopped and continued.

The sizes:
  * network: NETWORK, a card move transformer of depth 2, 4 heads, D = 128,
    FFN 2D, f32 matmuls (bf16 is slow on most CPUs), over the card planes
    encoders.schnaps_card_planes [70, 20].
  * rollout: NUM_ENVS (128) tables x NUM_STEPS (84) plies per update, every
    seat a perspective seat (512 batch rows, about 3 deals per table).
  * learner: one epoch of at least SGD_STEPS (8) minibatches of at most
    MINIBATCH (2,048) plies; the full recipe's loss and schedules, but the
    magnet temperature held at TEMPERATURE_FLOOR (0.015) or above.
  * length: NUM_UPDATES (2,000) updates; a keep every KEEP_EVERY (25).
  * in-run tracking evals: vs_uniform every UNIFORM_EVERY (50) updates on
    UNIFORM_DEALS (128) duplicate deals, and against Elena, the rule bot
    (envs.schnaps_elena), every ELENA_EVERY (50) on ELENA_DEALS (256).

Every in-run number is TRACKING tier.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.configs import (
    schnaps_move_transformer_self_play as full,
)
from open_spiel.python.scale.encoders import schnaps_card_planes as planes
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import ppo_magnet
from open_spiel.python.scale.networks import card_move_transformer

NAME = "schnaps_move_transformer_cpu"
NUM_ENVS = 128
NUM_STEPS = 84
NUM_UPDATES = 2000
KEEP_EVERY = 25
SGD_STEPS = 8
MINIBATCH = 2048
#: The most rows per learner forward; bounds the memory of a forward.
FORWARD_CHUNK = 2048
UNIFORM_EVERY, UNIFORM_DEALS = 50, 128
#: The Elena rule-bot eval (envs.schnaps_elena): every, deal pairs, seed.
ELENA_EVAL = "schnaps.elena"
ELENA_EVERY, ELENA_DEALS, ELENA_SEED = 50, 256, 4242
#: The magnet temperature's floor. The recipe's 0.001 suits 4,096 tables per
#: update; at 128 its decay (0.05 (n + 1)^-0.3) ends exploration after too
#: little play, and the bidding collapsed to passing. 0.015 keeps trying
#: calls and flecks all run long.
TEMPERATURE_FLOOR = 0.015
NETWORK = card_move_transformer.card_move_transformer(
    depth=2,
    embed_dim_per_head_over8=4,
    n_head=4,
    ff_factor=2,
    in_channels=planes.N_CHANNEL,
    n_card=planes.N_CARD,
    n_special_action=schnaps.NUM_ACTIONS - schnaps.NUM_CARDS,
    compute_dtype="float32",
)

NOTES = (
    "schnaps self-play at CPU size: a compact card move transformer on all "
    "four seats, ppo_magnet, team reward; in-run evals are TRACKING tier"
)


def policy_kwargs() -> dict[str, object]:
  """The network policy's arch_kwargs: NETWORK with the full recipe's options.

  Returns:
    The arch_kwargs.
  """
  return common.network_policy_kwargs(
      schnaps.SchnapsEnv.name,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      schnaps.NUM_ACTIONS,
      schnaps.NUM_PLAYERS,
      card_move_transformer.with_options(
          NETWORK, value_init=full.VALUE_INIT, center_keys=full.CENTER_KEYS
      ),
  )


def learner(num_updates: int = NUM_UPDATES):
  """The learner's PGConfig: the full recipe's loss at CPU batch sizes.

  Args:
    num_updates: the run length in updates.

  Returns:
    The pg.PGConfig.
  """
  return recipes.ppo_magnet_config(
      num_updates=num_updates,
      sgd_steps=SGD_STEPS,
      value_scale=full.VALUE_SCALE,
      minibatch_size=MINIBATCH,
      forward_chunk_rows=FORWARD_CHUNK,
      value_rows=full.VALUE_ROWS,
      max_decision_share=full.MAX_DECISION_SHARE,
      loss_config=ppo_magnet.PPOMagnetConfig(
          temperature_floor=TEMPERATURE_FLOOR
      ),
  )


def evals() -> tuple[runner.InLoopEval, ...]:
  """The in-run tracking evals: vs_uniform and vs Elena.

  Returns:
    The evals.
  """
  return full.evals(every=UNIFORM_EVERY, deals=UNIFORM_DEALS) + (
      runner.InLoopEval(
          ELENA_EVAL, every=ELENA_EVERY, deals=ELENA_DEALS, seed=ELENA_SEED
      ),
  )


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.ppo_magnet_self_play(
      NAME,
      schnaps.SchnapsEnv.name,
      learner(),
      policy_kwargs(),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=NUM_UPDATES,
      seed=full.SEED,
      keep_every=KEEP_EVERY,
      env_kwargs=dict(schnaps.OPTIONS),
      eval=evals(),
      notes=NOTES,
  )
