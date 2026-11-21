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

r"""Writes the step-4 network as the warm start of the mixed (step 2a) run.

  python -m open_spiel.python.scale.examples.schnaps_mixed_init

configs.schnaps_mixed_big_gpu warm-starts from the step-4 network
(~/schnaps_normal_big_gpu@1000), and a warm start must have the learner's
spec: the mixed env, and a final-score head widened from +-3 to +-12 points.
This builds the mixed config's policy, keeps its freshly initialised score
head params and takes every other param from the source, then writes the
result to the config's init_artifact() under the mixed env's rules identity
and EnvSpec, with the source as its parent. Apart from the score head, the
source and the new network must agree on every param's path, shape and
dtype, and their arch_kwargs may differ only in the game and the score head's
width. An existing target step is left alone.
"""

import argparse
import json
import os
import sys

import jax

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import schnaps_mixed_big_gpu as mixed
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import refs

#: The params that start fresh: the widened score head's.
FRESH = f"aux_{mixed.heads.SCORE}"


def _key(path) -> str:
  return "/".join(str(getattr(p, "key", p)) for p in path)


def _fresh(key: str) -> bool:
  return any(part.startswith(FRESH) for part in key.split("/"))


def merge(fresh, src):
  """`fresh` with every leaf outside the score head taken from `src`.

  Args:
    fresh: the new policy's params.
    src: the source params.

  Returns:
    The merged params (fresh's tree).

  Raises:
    ValueError: the trees differ outside the score head.
  """
  f = {_key(p): x for p, x in jax.tree_util.tree_flatten_with_path(fresh)[0]}
  s = {_key(p): x for p, x in jax.tree_util.tree_flatten_with_path(src)[0]}
  kept = {k for k in s if not _fresh(k)}
  if kept != {k for k in f if not _fresh(k)}:
    raise ValueError(
        "the networks differ outside the score head: "
        f"{sorted(kept ^ {k for k in f if not _fresh(k)})}"
    )
  for k in kept:
    if f[k].shape != s[k].shape or f[k].dtype != s[k].dtype:
      raise ValueError(
          f"param {k}: source {s[k].shape} {s[k].dtype}, new {f[k].shape} "
          f"{f[k].dtype}"
      )
  if not any(_fresh(k) for k in f):
    raise ValueError(f"the new network has no {FRESH} params")
  paths, treedef = jax.tree_util.tree_flatten_with_path(fresh)
  leaves = [x if _fresh(_key(p)) else s[_key(p)] for p, x in paths]
  print(f"kept {len(kept)} source params; fresh: "
        f"{sorted(k for k in f if _fresh(k))}")
  return jax.tree_util.tree_unflatten(treedef, leaves)


def _plain(x) -> str:
  return json.dumps(json.loads(json.dumps(x, sort_keys=True, default=str)),
                    sort_keys=True)


def _without_game_and_score(kw) -> dict:
  kw = json.loads(_plain(kw))
  kw.pop("game", None)
  kw["network"]["aux_heads"] = [
      h for h in kw["network"].get("aux_heads", ())
      if h["key"] != mixed.heads.SCORE
  ]
  return kw


def init(src_dir: str, src_step: int, dst: str) -> str:
  """Writes the warm start (module doc).

  Args:
    src_dir: the step-4 run's checkpoint dir.
    src_step: its step.
    dst: the target artifact id "<dir>@<step>".

  Returns:
    The target artifact id.

  Raises:
    ValueError: the source's network is not the mixed config's.
  """
  dst_dir, dst_step = dst.rsplit("@", 1)
  dst_step = int(dst_step)
  if artifact.exists(dst_dir, dst_step):
    print(f"{dst} exists; left alone")
    return dst
  src = artifact.load_artifact(src_dir, src_step)
  kwargs = mixed.policy_kwargs()
  if _plain(_without_game_and_score(src.manifest.arch_kwargs)) != _plain(
      _without_game_and_score(kwargs)
  ):
    raise ValueError(f"{src_dir}@{src_step}'s network is not the mixed config's")
  policy, fresh = refs.PolicyRef(
      arch_name=src.manifest.arch_name, arch_kwargs=kwargs
  ).build(jax.random.key(mixed.full.SEED))
  params = merge(fresh, src.params)
  env = env_registry.make(mixed.ENV)
  src_id = f"{artifact.norm_path(src_dir)}@{src_step}"
  artifact.write_artifact(
      dst_dir,
      dst_step,
      policy,
      params,
      rules_sha=env.declared_rules_sha,
      env_spec=env_registry.spec_of(env).to_json(),
      default_rule=src.manifest.default_rule,
      env_name=env.name,
      parents=(src_id,),
      lineage=artifact.lineage(src_id, src_step, mixed_init=mixed.NAME),
      extensions=dict(src.manifest.extensions),
  )
  print(f"wrote {dst} from {src_id}")
  return dst


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--src_dir", default=mixed.SRC_DIR)
  p.add_argument("--src_step", type=int, default=mixed.INIT_STEP)
  p.add_argument("--dst", default="", help="default: the config's init")
  a = p.parse_args(argv)
  x64.enable_x64()
  dst = a.dst or mixed.init_artifact()
  if not os.path.isdir(os.path.expanduser(a.src_dir)):
    sys.exit(f"no checkpoint dir {a.src_dir}")
  init(os.path.expanduser(a.src_dir), a.src_step, dst)


if __name__ == "__main__":
  main()
