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

r"""Schnaps card play with Schnapser forced, round 2 (new Schnapser rule).

The recipe is configs.schnaps_forced_cpu's config_round2: a further
fine-tune of ~/schnaps_forced_schnapser@1000 after Schnapser became "win
each of the first three tricks".

  python -m open_spiel.python.scale.train \
      --config=schnaps_forced_schnapser_r2_cpu \
      --checkpoint_dir=$HOME/schnaps_forced_schnapser_r2
"""

from open_spiel.python.scale.configs import schnaps_forced_cpu as forced
from open_spiel.python.scale.league import runner

CONTRACT = "schnapser"
NAME = "schnaps_forced_schnapser_r2_cpu"


def get_config() -> runner.LeagueRunConfig:
  """The run config (configs.schnaps_forced_cpu.config_round2).

  Returns:
    The runner.LeagueRunConfig.
  """
  return forced.config_round2(CONTRACT)
