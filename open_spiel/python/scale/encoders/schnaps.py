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

"""Schnaps encoder over the envs.schnaps InfoState.

``schnaps_info_state`` is the games package's schnaps information-state
tensor (its module doc has the layout), rebuilt from one seat's InfoState;
it equals the game's own tensor on every seat of random deals
(envs/tests/schnaps_test.py). OpenSpiel's schnaps has no tensor, so this is
the package's own encoding, checked there to carry exactly what OpenSpiel's
information-state string carries. Its length depends on ``max_flecks`` (the
bid slots); the registered shape is the default's.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import schnaps

F32 = jnp.float32
INFO_STATE_NAME = "schnaps_info_state"


def info_state(info) -> jax.Array:
  """The games package's schnaps information-state tensor (module doc).

  Args:
    info: one seat's InfoState (envs.schnaps).

  Returns:
    f32[game.info_state_size(max_flecks)].
  """
  return game.info_state_tensor(
      seat=info["seat"],
      phase=info["phase"],
      first_cards=info["first_cards"],
      dealt=info["dealt"],
      hand=info["hand"],
      trump=info["trump"],
      team_score=info["team_score"],
      pending=info["pending"],
      team_tricks=info["team_tricks"],
      bids=info["bids"],
      plays=info["plays"],
      declared=info["declared"],
  )


INFO_STATE = registry.register(
    registry.Encoder(
        name=INFO_STATE_NAME,
        game=schnaps.SchnapsEnv.name,
        shape=(game.info_state_size(game.DEFAULT_MAX_FLECKS),),
        dtype=F32,
        fn=info_state,
        reference=(
            "games package schnaps information_state_tensor (OpenSpiel's "
            "schnaps has none)"
        ),
    )
)
