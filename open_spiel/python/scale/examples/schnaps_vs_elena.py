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

r"""Score schnaps checkpoints against Elena, the rule bot.

  python -m open_spiel.python.scale.examples.schnaps_vs_elena \
      --checkpoint_dir=$HOME/schnaps_cpu                      # the latest
  python -m open_spiel.python.scale.examples.schnaps_vs_elena \
      --checkpoint_dir=$HOME/schnaps_cpu --steps=100,300,500 --deals=512
  python -m open_spiel.python.scale.examples.schnaps_vs_elena \
      --checkpoint_dir=$HOME/schnaps_cpu --steps=all --opponent=uniform

The checkpoint's partnership (seats 0 and 2) plays the opponent's (seats 1
and 3) on --deals duplicate deals, each played twice with the partnerships
swapped, so the luck of the cards cancels. The checkpoint plays greedily
(its most likely action), as the in-run evals do. Printed per step: the
checkpoint's mean points per deal with its standard error and 95% interval;
positive means it beats the opponent. The deals are the same for every step
(--seed), so steps compare fairly.

--opponent=uniform plays the random opponent of the in-run vs_uniform eval;
--baseline also prints Elena against the random opponent, for scale.
"""

import argparse
import os
import sys
import time

import jax

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import schnaps_elena as elena
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.policies import uniform


def _steps(path: str, spec: str) -> list[int]:
  """The steps `spec` names: "latest", "all", or a comma-separated list.

  Args:
    path: the checkpoint dir.
    spec: the --steps value.

  Returns:
    The steps, ascending.

  Raises:
    SystemExit: a step not held, or no step at all.
  """
  held = artifact.steps(path)
  if not held:
    sys.exit(f"no checkpoint under {path}")
  if spec == "latest":
    return [held[-1]]
  if spec == "all":
    return list(held)
  want = sorted({int(s) for s in spec.split(",") if s.strip()})
  missing = [s for s in want if s not in held]
  if missing:
    sys.exit(f"steps {missing} not in {path}; held: {list(held)}")
  return want


def _uniform_entrant(env) -> match.Entrant:
  """The uniform-random opponent.

  Args:
    env: the env.

  Returns:
    The Entrant.
  """
  pol = uniform.UniformPolicy(
      env.name, int(env.num_actions), int(env.num_players)
  )
  return match.Entrant("uniform", pol, {}, types.SAMPLE)


def _line(label: str, res) -> str:
  """One result line.

  Args:
    label: what played.
    res: the MatchResult.

  Returns:
    The line.
  """
  m = res.mean
  return (
      f"{label:>12}  {m.mean:+8.2f} +- {m.se:5.2f}   "
      f"95% [{m.lo:+7.2f}, {m.hi:+7.2f}]"
  )


def main(argv=None) -> None:
  """Scores the checkpoints (module doc).

  Args:
    argv: the arguments (default sys.argv[1:]).
  """
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--checkpoint_dir", default=os.path.expanduser("~/schnaps_cpu"))
  p.add_argument(
      "--steps", default="latest", help='"latest", "all", or e.g. 100,300,500'
  )
  p.add_argument("--deals", type=int, default=256, help="duplicate deal pairs")
  p.add_argument("--opponent", choices=("elena", "uniform"), default="elena")
  p.add_argument("--baseline", action="store_true")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args(argv)

  x64.enable_x64()
  path = os.path.expanduser(a.checkpoint_dir)
  steps = _steps(path, a.steps)
  first = artifact.load_artifact(path, steps[0])
  env = env_registry.training_env(first.manifest)
  ev = elena.evaluator(env)
  opponent = elena.entrant() if a.opponent == "elena" else _uniform_entrant(env)
  key = jax.random.key(a.seed)
  print(
      f"{env.name} {dict(env_registry.spec_of(env).kwargs)}: {a.deals} deal "
      f"pairs against {opponent.name}; points per deal, checkpoint's side"
  )
  if a.baseline:
    res = ev.head_to_head(elena.entrant(), _uniform_entrant(env), a.deals, key)
    print(_line("elena/unif", res))
  for step in steps:
    art = first if step == steps[0] else artifact.load_artifact(path, step)
    subject = match.Entrant.from_artifact(
        art, rule=types.ARGMAX, name=f"step{step}"
    )
    t0 = time.time()
    res = ev.head_to_head(subject, opponent, a.deals, key)
    print(_line(f"step {step}", res) + f"   ({time.time() - t0:.0f}s)")
    sys.stdout.flush()


if __name__ == "__main__":
  main()
