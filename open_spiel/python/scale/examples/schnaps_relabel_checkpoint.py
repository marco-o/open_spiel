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

r"""Re-saves the Normal-only network as the warm start of a forced-contract run.

  python -m open_spiel.python.scale.examples.schnaps_relabel_checkpoint \
      --contract=lomt

A network's spec names the env it was trained on, and a warm start must have
the learner's spec (league.kinds.pg), so the Normal-only checkpoint cannot
start a schnaps_forced_<contract> run as it is. This writes the same params
under the forced env's spec (configs.schnaps_forced_cpu.policy_kwargs: the
same network and encoder, only `game` differs) to
configs.schnaps_forced_cpu.init_artifact(contract), with the forced env's
rules identity and EnvSpec and the source as its parent. The params are
checked to match the new policy's tree, shapes and dtypes; nothing else of
the source changes. An existing target step is left alone.
"""

import argparse
import os
import sys

import jax

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import schnaps_forced_cpu as forced
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import refs


def relabel(
    contract: str,
    src_dir: str,
    src_step: int,
    dst: str,
) -> str:
  """Writes the relabelled artifact (module doc).

  Args:
    contract: "schnapser", "lomt" or "bauern".
    src_dir: the Normal-only run's checkpoint dir.
    src_step: its step.
    dst: the target artifact id "<dir>@<step>".

  Returns:
    The target artifact id.

  Raises:
    ValueError: the source's network is not the forced config's.
  """
  dst_dir, dst_step = dst.rsplit("@", 1)
  dst_step = int(dst_step)
  if artifact.exists(dst_dir, dst_step):
    print(f"{dst} exists; left alone")
    return dst
  src = artifact.load_artifact(src_dir, src_step)
  kwargs = forced.policy_kwargs(contract)
  old = dict(src.manifest.arch_kwargs)
  if {k: v for k, v in old.items() if k != "game"} != {
      k: v for k, v in kwargs.items() if k != "game"
  }:
    raise ValueError(
        f"{src_dir}@{src_step}'s network is not configs.schnaps_forced_cpu's"
    )
  policy, fresh = refs.PolicyRef(
      arch_name=src.manifest.arch_name, arch_kwargs=kwargs
  ).build(jax.random.key(0))
  same = jax.tree_util.tree_structure(fresh) == jax.tree_util.tree_structure(
      src.params
  ) and all(
      a.shape == b.shape and a.dtype == b.dtype
      for a, b in zip(
          jax.tree_util.tree_leaves(fresh), jax.tree_util.tree_leaves(src.params)
      )
  )
  if not same:
    raise ValueError("the source params do not fit the forced env's network")
  env = env_registry.make(forced.env_name(contract))
  src_id = f"{artifact.norm_path(src_dir)}@{src_step}"
  artifact.write_artifact(
      dst_dir,
      dst_step,
      policy,
      src.params,
      rules_sha=env.declared_rules_sha,
      env_spec=env_registry.spec_of(env).to_json(),
      default_rule=src.manifest.default_rule,
      env_name=env.name,
      parents=(src_id,),
      lineage=artifact.lineage(src_id, src_step, relabelled_for=env.name),
      extensions=dict(src.manifest.extensions),
  )
  print(f"wrote {dst} from {src_id} for {env.name}")
  return dst


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--contract", required=True, choices=forced.CONTRACTS)
  p.add_argument("--src_dir", default=forced.NORMAL_DIR)
  p.add_argument("--src_step", type=int, default=forced.INIT_STEP)
  p.add_argument("--dst", default="", help="default: the config's init")
  a = p.parse_args(argv)
  x64.enable_x64()
  dst = a.dst or forced.init_artifact(a.contract)
  if not os.path.isdir(os.path.expanduser(a.src_dir)):
    sys.exit(f"no checkpoint dir {a.src_dir}")
  relabel(a.contract, os.path.expanduser(a.src_dir), a.src_step, dst)


if __name__ == "__main__":
  main()
