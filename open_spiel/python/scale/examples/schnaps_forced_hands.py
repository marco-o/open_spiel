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

r"""Writes the forced-contract hands file from the oracle's solves.

  python -m open_spiel.python.scale.examples.schnaps_forced_hands \
      --samples=40 --workers=4

For each forced contract (envs.schnaps_forced) and each hand class, at each
declarer seat (one seat for Lomt, where the declarer leads, so every seat is
alike; four for Schnapser and Bauern, where seat 0 leads), the declarer holds
a hand of the class (trump in suit 0) and the other fifteen cards are dealt
at random --samples times. The oracle (envs.schnaps_oracle) solves each deal
with every hand visible; p is the share the declarer wins. It is biased (the
defence never errs and nobody guesses), which is good enough to choose
training hands. Each class's weight (schnaps_forced.class_weights): 1 when
p >= --hi, a constant for --lo <= p < --hi so those weak hands are
--weak_share of the dealt hands, 0 below --lo.

Deterministic for a given --seed. The default output is the file the envs
read (envs/schnaps_forced_hands.json); --contracts and --max_classes make
small files for tests.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

from open_spiel.python.scale.envs import schnaps_forced as sf
from open_spiel.python.scale.envs import schnaps_oracle as oracle


def seats_of(contract: int) -> tuple[int, ...]:
  """The declarer seats estimated: one for Lomt, four otherwise."""
  return (0,) if contract == oracle.LOMT else tuple(range(oracle.N_PLAYER))


def success_rate(job: tuple[int, int, tuple[int, ...], int, int, int]) -> float:
  """The oracle's win rate for one class at one declarer seat.

  Args:
    job: (contract, class index, class key, seat, samples, seed).

  Returns:
    The share of deals the declarer wins.
  """
  contract, ci, key, seat, samples, seed = job
  rng = np.random.default_rng([seed, contract, ci, seat])
  mine = sf.class_hand(key)
  rest = np.array([c for c in range(oracle.N_CARD) if c not in mine])
  others = [s for s in range(oracle.N_PLAYER) if s != seat]
  rules = oracle.Rules(0, contract, seat)
  side = 1 if seat % 2 == 0 else -1
  wins = 0
  for _ in range(samples):
    deal = rng.permutation(rest)
    hands = [0] * oracle.N_PLAYER
    for c in mine:
      hands[seat] |= 1 << c
    for j, c in enumerate(deal):
      hands[others[j // oracle.HAND_SIZE]] |= 1 << int(c)
    v = oracle.PlaySolver(rules).value(oracle.initial_play(tuple(hands), rules))
    wins += side * v > 0
  return wins / samples


def contract_entry(
    name: str,
    *,
    samples: int,
    seed: int,
    lo: float,
    hi: float,
    weak_share: float,
    max_classes: int,
    pool,
) -> dict:
  """One contract's entry of the hands file.

  Args:
    name: the contract ("schnapser", "lomt", "bauern").
    samples: deals per class and seat.
    seed: the seed.
    lo: class_weights's lo.
    hi: class_weights's hi.
    weak_share: class_weights's weak_share.
    max_classes: 0, or estimate only this many classes (the others get p 0;
      for tests).
    pool: a multiprocessing pool, or None.

  Returns:
    {"contract", "trump_marked", "seats", "classes", "p", "weight"}.
  """
  contract = sf.CONTRACTS[name]
  marked = sf.trump_marked_of(contract)
  classes, _ = sf.hand_classes(marked)
  seats = seats_of(contract)
  n = len(classes) if not max_classes else min(max_classes, len(classes))
  jobs = [
      (contract, ci, classes[ci], seat, samples, seed)
      for ci in range(n)
      for seat in seats
  ]
  t0 = time.time()
  rates = (pool.map if pool else map)(success_rate, jobs, *([16] if pool else []))
  p = np.zeros((len(classes), len(seats)))
  p[:n] = np.asarray(list(rates)).reshape(n, len(seats))
  counts = sf.class_counts(marked)
  w = np.stack(
      [
          sf.class_weights(
              p[:, j], counts, lo=lo, hi=hi, weak_share=weak_share
          )
          for j in range(len(seats))
      ],
      axis=1,
  )
  kept = (w > 0).any(axis=1).sum()
  print(
      f"{name}: {n} of {len(classes)} classes x {len(seats)} seats x "
      f"{samples} deals in {time.time() - t0:.0f}s; mean p "
      f"{float((p * counts[:, None]).sum() / counts.sum() / len(seats)):.3f} "
      f"over all hands; {kept} classes kept",
      flush=True,
  )
  if len(seats) == 1:
    p, w = p[:, 0], w[:, 0]
  return {
      "contract": contract,
      "trump_marked": marked,
      "seats": list(seats),
      "classes": [list(k) for k in classes],
      "p": np.round(p, 4).tolist(),
      "weight": np.round(w, 6).tolist(),
  }


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--samples", type=int, default=40)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--lo", type=float, default=0.02)
  p.add_argument("--hi", type=float, default=0.15)
  p.add_argument("--weak_share", type=float, default=0.15)
  p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
  p.add_argument("--contracts", default=",".join(sf.CONTRACTS))
  p.add_argument("--max_classes", type=int, default=0)
  p.add_argument("--out", default=sf.HANDS_FILE)
  a = p.parse_args(argv)
  names = [c for c in a.contracts.split(",") if c]
  unknown = sorted(set(names) - set(sf.CONTRACTS))
  if unknown:
    sys.exit(f"unknown contracts {unknown}; known: {sorted(sf.CONTRACTS)}")
  out = {
      "schema": sf.HANDS_SCHEMA,
      "params": {
          "samples": a.samples,
          "seed": a.seed,
          "lo": a.lo,
          "hi": a.hi,
          "weak_share": a.weak_share,
          "max_classes": a.max_classes,
      },
      "contracts": {},
  }
  pool = (
      multiprocessing.get_context("spawn").Pool(a.workers)
      if a.workers > 1
      else None
  )
  try:
    for name in names:
      out["contracts"][name] = contract_entry(
          name,
          samples=a.samples,
          seed=a.seed,
          lo=a.lo,
          hi=a.hi,
          weak_share=a.weak_share,
          max_classes=a.max_classes,
          pool=pool,
      )
  finally:
    if pool:
      pool.close()
      pool.join()
  with open(a.out, "w") as f:
    json.dump(out, f, separators=(",", ":"))
    f.write("\n")
  print(f"wrote {a.out}")


if __name__ == "__main__":
  main()
