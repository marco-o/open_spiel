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

r"""Schnaps search bots against Elena: the oracle (ceiling) and the fair one.

  python open_spiel/python/scale/examples/schnaps_oracle_vs_elena.py \
      --subject oracle --deals 256
  python open_spiel/python/scale/examples/schnaps_oracle_vs_elena.py \
      --subject search --samples 20 --deals 256 --workers 4

Duplicate deals: each deal is played twice with the same cards in the same
seats, the subject's partnership on seats {0, 2} and then on {1, 3}; a
pair's score is the mean of its two deals, in game points per deal for the
subject's partnership (the same unit as the in-run "schnaps.elena/mean").

Subjects:
  oracle  envs.schnaps_oracle: sees every hand and solves the bidding and
          the play exactly (`--phases` limits where; Elena elsewhere).
  search  envs.schnaps_search: fair; samples the hidden hands `--samples`
          times per decision and solves each. `--phases play` (the default
          for search) leaves the bidding to Elena's rules.
  elena   Elena against Elena through the same loop: a check, about 0.
The trump choice, made before the hands exist, is always Elena's.
"""

import argparse
import collections
import math
import multiprocessing
import sys
import time

import numpy as np

from open_spiel.python.scale.envs import schnaps_oracle as oracle
from open_spiel.python.scale.envs import schnaps_search as search

CONTRACT_NAMES = ("Normal", "Schnapser", "Lomt", "Bauern", "Jodler", "Hermort")
PHASES = {
    "all": (oracle.BIDDING, oracle.PLAY),
    "play": (oracle.PLAY,),
    "bid": (oracle.BIDDING,),
}


def _parse(argv):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--subject", choices=("oracle", "search", "elena"),
                 default="oracle")
  p.add_argument("--deals", type=int, default=256, help="deal pairs")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--max_flecks", type=int, default=2)
  p.add_argument(
      "--phases",
      "--oracle_phases",
      choices=tuple(PHASES),
      default=None,
      help="where the subject decides, Elena's rules elsewhere (default: "
      "all for oracle, play for search)",
  )
  p.add_argument("--samples", type=int, default=20,
                 help="search: sampled deals per decision")
  p.add_argument("--workers", type=int, default=1,
                 help="processes, each playing a share of the deal pairs")
  args = p.parse_args(argv)
  if args.phases is None:
    args.phases = "play" if args.subject == "search" else "all"
  return args


class ElenaPlayer:
  """Elena's rules (envs.schnaps_elena) on a Model's seat view."""

  def __init__(self):
    # JAX is imported here, so a worker process loads it itself.
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from open_spiel.python.scale.core import x64  # pylint: disable=g-import-not-at-top
    from open_spiel.python.scale.envs import schnaps_elena as elena  # pylint: disable=g-import-not-at-top
    x64.enable_x64()
    self._jax, self._jnp = jax, jnp
    self._act = jax.jit(elena.action)
    self._key = jax.random.key(0)
    self._n = 0

  def reseed(self, seed: int) -> None:
    self._key = self._jax.random.key(seed)
    self._n = 0

  def __call__(self, m: oracle.Model) -> int:
    self._n += 1
    jnp = self._jnp
    info = {k: jnp.asarray(v) for k, v in m.info_state(m.current_player()).items()}
    key = self._jax.random.fold_in(self._key, self._n)
    return int(self._act(info, jnp.asarray(m.legal_mask()), key))


def play_deal(perm, subject_team, args, fair, seed):
  """One deal; returns (the subject partnership's return, the final Model)."""
  m = oracle.Model(max_flecks=args.max_flecks)
  deck = iter(int(c) for c in perm)
  phases = PHASES[args.phases]
  solver = None
  bot = None
  if args.subject == "search":
    bot = search.SearchBot(args.samples, seed, phases, args.max_flecks)
  while not m.is_terminal():
    if m.is_chance():
      m.apply(next(deck))
      continue
    a = None
    if oracle.team_of(m.current_player()) == subject_team:
      if args.subject == "oracle" and m.phase in phases:
        if solver is None:
          solver = oracle.Oracle(m.hands(), m.trump, args.max_flecks)
        a = solver.decide(m)
      elif bot is not None:
        a = bot.decide(m)
    if a is None:
      a = fair(m)
    m.apply(a)
  return m.returns[subject_team], m


_FAIR = None


def play_pair(job):
  """Deal pair i, both orientations: (pair score, per-deal facts)."""
  global _FAIR
  args, i = job
  if _FAIR is None:
    _FAIR = ElenaPlayer()
  perm = np.random.default_rng([args.seed, i]).permutation(oracle.N_CARD)
  scores, facts = [], []
  for subject_team in (0, 1):
    _FAIR.reseed(args.seed * 1_000_003 + 2 * i + subject_team)
    r, m = play_deal(perm, subject_team, args, _FAIR,
                     seed=args.seed * 1_000_003 + 2 * i + subject_team)
    scores.append(r)
    side = None
    if m.contract:
      side = "subject" if oracle.team_of(m.declarer) == subject_team else "fair"
    facts.append((CONTRACT_NAMES[m.contract], side, m.flecks > 0))
  return sum(scores) / 2, facts


def _summary(x):
  se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
  return x.mean(), se


def main(argv=None):
  args = _parse(argv)
  jobs = [(args, i) for i in range(args.deals)]
  pair_scores = []
  contracts = collections.Counter()
  called_by = collections.Counter()
  flecked = 0
  start = time.time()
  if args.workers > 1:
    pool = multiprocessing.get_context("spawn").Pool(args.workers)
    results = pool.imap(play_pair, jobs)
  else:
    pool = None
    results = map(play_pair, jobs)
  for score, facts in results:
    pair_scores.append(score)
    for name, side, fl in facts:
      contracts[name] += 1
      if side:
        called_by[side] += 1
      flecked += fl
    k = len(pair_scores)
    if k % 32 == 0 or k == args.deals:
      mean, se = _summary(np.asarray(pair_scores))
      print(f"{k:5d} pairs  mean {mean:+.3f} +- {se:.3f}  "
            f"({time.time() - start:.0f}s)", flush=True)
  if pool is not None:
    pool.close()
  mean, se = _summary(np.asarray(pair_scores))
  n = 2 * args.deals
  what = args.subject
  if args.subject != "elena":
    what += f" ({args.phases})"
  if args.subject == "search":
    what += f", {args.samples} samples"
  print()
  print(f"{what} vs elena over {args.deals} duplicate deal pairs:")
  print(f"  mean {mean:+.3f} points per deal, se {se:.3f}, "
        f"95% [{mean - 1.96 * se:+.3f}, {mean + 1.96 * se:+.3f}]")
  print("  contracts played: " + ", ".join(
      f"{k} {v / n:.1%}" for k, v in contracts.most_common()))
  print(f"  contracts called by the subject side: {called_by['subject']}, "
        f"by Elena's side: {called_by['fair']}; deals with flecks: {flecked}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
