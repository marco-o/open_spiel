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

r"""Writes a full schnaps player (envs.schnaps_player) as an artifact.

  python -m open_spiel.python.scale.examples.schnaps_build_player \
      --step=0                                   # greedy bidder
  python -m open_spiel.python.scale.examples.schnaps_build_player \
      --step=1 --rule=pass                       # the Normal-only baseline
  python -m open_spiel.python.scale.examples.schnaps_build_player \
      --step=2 --margin=1

It loads the Normal network (--normal, default ~/schnaps_normal@4000) and
the forced-contract networks (--forced_dir_prefix, default
~/schnaps_forced_<contract>@1000), checks that they are one architecture
(only the env name differs), and writes the player to --out_dir/<step>
(default ~/schnaps_player) for schnaps(max_flecks=2), with the four sources
as parents. Steps here only label variants. The artifact plays wherever a
checkpoint does: examples/schnaps_vs_elena.py, the GUI's play launcher.
An existing step is replaced only with --overwrite.
"""

import argparse
import os
import shutil
import sys

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import schnaps_player as player_lib


def _load(ckpt: str):
  path, step = ckpt.rsplit("@", 1)
  path = os.path.expanduser(path)
  return f"{artifact.norm_path(path)}@{step}", artifact.load_artifact(
      path, int(step)
  )


def build(
    sources: dict[str, str],
    out_dir: str,
    step: int,
    rule: str,
    rule_kwargs: dict,
    overwrite: bool = False,
) -> str:
  """Writes the player (module doc).

  Args:
    sources: {part name: "<dir>@<step>"}, player_lib.PART_NAMES.
    out_dir: the output dir.
    step: the output step.
    rule: the bid rule.
    rule_kwargs: its kwargs.
    overwrite: replace an existing step.

  Returns:
    The artifact id.

  Raises:
    ValueError: the sources are not one architecture.
  """
  out_dir = os.path.expanduser(out_dir)
  target = os.path.join(out_dir, str(step))
  if artifact.exists(out_dir, step):
    if not overwrite:
      raise ValueError(f"{out_dir}@{step} exists (use --overwrite)")
    shutil.rmtree(target)
  loaded = {n: _load(sources[n]) for n in player_lib.PART_NAMES}
  ref = None
  scales = set()
  for n, (src_id, art) in loaded.items():
    kw = {k: v for k, v in art.manifest.arch_kwargs.items() if k != "game"}
    if ref is None:
      ref = (art.manifest.arch_name, kw)
    elif (art.manifest.arch_name, kw) != ref:
      raise ValueError(f"{src_id} ({n}) is not the same network as the others")
    scales.add(float(art.manifest.extensions.get("value_scale", 1.0)))
  if len(scales) != 1:
    raise ValueError(f"the sources disagree on value_scale: {sorted(scales)}")
  value_scale = scales.pop()
  policy = player_lib.SchnapsPlayerPolicy(
      parts={n: art.policy for n, (_, art) in loaded.items()},
      rule=rule,
      rule_kwargs=rule_kwargs,
      value_scale=value_scale,
  )
  params = {n: art.params for n, (_, art) in loaded.items()}
  env = env_registry.make("schnaps")
  normal = loaded["normal"][1]
  artifact.write_artifact(
      out_dir,
      step,
      policy,
      params,
      rules_sha=env.declared_rules_sha,
      env_spec=env_registry.spec_of(env).to_json(),
      default_rule=normal.manifest.default_rule,
      env_name=env.name,
      parents=tuple(src_id for src_id, _ in loaded.values()),
      lineage=artifact.lineage(
          loaded["normal"][0],
          normal.manifest.step,
          assembled_from={n: src_id for n, (src_id, _) in loaded.items()},
          rule=rule,
          rule_kwargs=dict(rule_kwargs),
      ),
      extensions={"value_scale": value_scale},
  )
  out = f"{artifact.norm_path(out_dir)}@{step}"
  print(f"wrote {out}: rule {rule} {rule_kwargs}, parts:")
  for n, (src_id, _) in loaded.items():
    print(f"  {n}: {src_id}")
  return out


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--normal", default="~/schnaps_normal@4000")
  p.add_argument("--forced_dir_prefix", default="~/schnaps_forced_")
  p.add_argument("--forced_step", type=int, default=1000)
  p.add_argument("--out_dir", default="~/schnaps_player")
  p.add_argument("--step", type=int, default=0)
  p.add_argument(
      "--rule", default="greedy",
      choices=player_lib.bidding.registered_rules(),
  )
  p.add_argument("--margin", type=float, default=None, help="greedy only")
  p.add_argument("--overwrite", action="store_true")
  a = p.parse_args(argv)
  x64.enable_x64()
  sources = {"normal": a.normal}
  for n in player_lib.PART_NAMES[1:]:
    sources[n] = f"{a.forced_dir_prefix}{n}@{a.forced_step}"
  rule_kwargs = {}
  if a.margin is not None:
    if a.rule != "greedy":
      sys.exit("--margin is a greedy option")
    rule_kwargs["margin"] = a.margin
  try:
    build(sources, a.out_dir, a.step, a.rule, rule_kwargs, a.overwrite)
  except ValueError as e:
    sys.exit(str(e))


if __name__ == "__main__":
  main()
