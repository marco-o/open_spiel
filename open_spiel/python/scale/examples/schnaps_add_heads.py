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

r"""Writes the step-1 GPU network with the step-3 aux heads added.

  python -m open_spiel.python.scale.examples.schnaps_add_heads

configs.schnaps_normal_heads_gpu warm-starts from the step-1 network
(~/schnaps_normal_gpu@2700) with two aux heads that network does not have,
and a warm start must have the learner's spec. This builds the heads
config's policy, keeps its freshly initialised aux head params and takes
every other param from the source, then writes the result to the config's
init_artifact() with the source as its parent. Every source param must have
the same path, shape and dtype in the new tree, and the new tree may add only
aux_<key> params. An existing target step is left alone.
"""

import argparse
import json
import os
import sys

import jax

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import schnaps_normal_heads_gpu as heads
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import refs


def _flat(tree) -> dict[str, jax.Array]:
  """{"a/b/c": leaf} of a params tree."""
  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    out["/".join(str(getattr(k, "key", k)) for k in path)] = leaf
  return out


def merge(fresh, src):
  """`fresh` with every leaf `src` has taken from `src` (module doc).

  Args:
    fresh: the new policy's params.
    src: the source params.

  Returns:
    The merged params (fresh's tree).

  Raises:
    ValueError: a source leaf is missing or differs in the new tree, or the
      new tree adds a leaf outside the aux heads.
  """
  f, s = _flat(fresh), _flat(src)
  for k, v in s.items():
    if k not in f:
      raise ValueError(f"source param {k} is not in the new network")
    if f[k].shape != v.shape or f[k].dtype != v.dtype:
      raise ValueError(
          f"param {k}: source {v.shape} {v.dtype}, new {f[k].shape} "
          f"{f[k].dtype}"
      )
  added = sorted(set(f) - set(s))
  bad = [k for k in added if not any(p.startswith("aux_") for p in k.split("/"))]
  if bad:
    raise ValueError(f"the new network adds params outside its aux heads: {bad}")
  paths, treedef = jax.tree_util.tree_flatten_with_path(fresh)
  leaves = []
  for path, leaf in paths:
    k = "/".join(str(getattr(p, "key", p)) for p in path)
    leaves.append(s.get(k, leaf))
  print(f"kept {len(s)} source params; new: {added}")
  return jax.tree_util.tree_unflatten(treedef, leaves)


def add_heads(src_dir: str, src_step: int, dst: str) -> str:
  """Writes the network with the heads added (module doc).

  Args:
    src_dir: the step-1 run's checkpoint dir.
    src_step: its step.
    dst: the target artifact id "<dir>@<step>".

  Returns:
    The target artifact id.

  Raises:
    ValueError: the source's network is not the heads config's body.
  """
  dst_dir, dst_step = dst.rsplit("@", 1)
  dst_step = int(dst_step)
  if artifact.exists(dst_dir, dst_step):
    print(f"{dst} exists; left alone")
    return dst
  src = artifact.load_artifact(src_dir, src_step)
  kwargs = heads.policy_kwargs()
  old = dict(src.manifest.arch_kwargs)
  body = dict(kwargs, network={
      k: v for k, v in kwargs["network"].items() if k != "aux_heads"
  })
  def plain(x):
    return json.dumps(x, sort_keys=True, default=str)

  if plain(json.loads(plain(old))) != plain(json.loads(plain(body))):
    raise ValueError(
        f"{src_dir}@{src_step}'s network is not the heads config's body"
    )
  policy, fresh = refs.PolicyRef(
      arch_name=src.manifest.arch_name, arch_kwargs=kwargs
  ).build(jax.random.key(heads.full.SEED))
  params = merge(fresh, src.params)
  env = env_registry.make(heads.ENV)
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
      lineage=artifact.lineage(src_id, src_step, added_heads=heads.NAME),
      extensions=dict(src.manifest.extensions),
  )
  print(f"wrote {dst} from {src_id}")
  return dst


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--src_dir", default=heads.STEP1_DIR)
  p.add_argument("--src_step", type=int, default=heads.INIT_STEP)
  p.add_argument("--dst", default="", help="default: the config's init")
  a = p.parse_args(argv)
  x64.enable_x64()
  dst = a.dst or heads.init_artifact()
  if not os.path.isdir(os.path.expanduser(a.src_dir)):
    sys.exit(f"no checkpoint dir {a.src_dir}")
  add_heads(os.path.expanduser(a.src_dir), a.src_step, dst)


if __name__ == "__main__":
  main()
