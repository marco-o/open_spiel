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

r"""Summarises round-2 measurements and writes them as a readable table.

  python -m open_spiel.python.scale.examples.schnaps_forced_report \
      ~/schnaps_forced_measure_lomt.json ... --csv=forced-hands-net.csv

Per contract (examples/schnaps_forced_measure output) it prints:
  * the network's success rate as declarer, per seat, over the measured
    classes (each class weighted by how often it is dealt);
  * the network's success against the oracle's p, by band of the oracle's p;
  * the value head's calibration: deals binned by the chance the value head
    predicts at the first play (p = (E / points + 1) / 2), against how often
    those deals were won.
The CSV has one row per measured (class, declarer seat): the hand in card
notation, the seat, the deals, the network's success rate, the value head's
predicted chance and the oracle's p.
"""

import argparse
import csv
import json

import numpy as np

from open_spiel.python.scale.envs import schnaps_forced as sf

RANKS = ("U", "O", "K", "10", "A")


def hand_text(key, marked: bool) -> str:
  """A class in card notation, e.g. "Trump: A 10 | K O U"."""

  def suit(m):
    return " ".join(RANKS[r] for r in range(4, -1, -1) if m >> r & 1)

  parts = []
  for i, m in enumerate(key):
    if marked and i == 0:
      parts.append("Trump: " + (suit(m) or "-"))
    elif m:
      parts.append(suit(m))
  return " | ".join(parts)


def summary(res: dict) -> list[str]:
  """The printed summary of one contract (module doc)."""
  name = res["contract"]
  marked = sf.trump_marked_of(sf.CONTRACTS[name])
  points = float(res["points"])
  n = np.asarray(res["n"], np.float64)
  wins = np.asarray(res["wins"], np.float64)
  vsum = np.asarray(res["value_sum"], np.float64)
  po = np.asarray(res["p_oracle"], np.float64)
  counts = sf.class_counts(marked).astype(np.float64)[:, None]
  has = n > 0
  p_net = np.where(has, wins / np.maximum(n, 1), np.nan)
  p_val = np.where(
      has, np.clip((vsum / np.maximum(n, 1) / points + 1) / 2, 0, 1), np.nan
  )
  lines = [f"== {name} ({int(points)} points): {int(n.sum())} deals"]
  for seat in range(sf.NUM_PLAYERS):
    m = has[:, seat]
    if not m.any():
      continue
    w = counts[m, 0]
    lines.append(
        f"  seat {seat}: network {np.average(p_net[m, seat], weights=w):.3f}"
        f"   oracle {np.average(po[m, seat], weights=w):.3f}"
        f"   value head {np.average(p_val[m, seat], weights=w):.3f}"
        "   (classes weighted by how often they are dealt)"
    )
  lines.append("  by the oracle's p:   oracle -> network   (deals)")
  for lo, hi in ((0.0, 0.15), (0.15, 0.4), (0.4, 0.7), (0.7, 0.95), (0.95, 1.01)):
    m = has & (po >= lo) & (po < hi)
    if n[m].sum() == 0:
      continue
    lines.append(
        f"    [{lo:.2f}, {min(hi, 1):.2f})   "
        f"{np.average(po[m], weights=n[m]):.3f} -> "
        f"{wins[m].sum() / n[m].sum():.3f}   ({int(n[m].sum())})"
    )
  # Calibration per cell (cell-level predicted chance vs win rate).
  lines.append("  value head: predicted -> won   (deals)")
  for lo in np.arange(0.0, 1.0, 0.2):
    m = has & (p_val >= lo) & (p_val < lo + 0.2 + (lo >= 0.8) * 0.01)
    if n[m].sum() == 0:
      continue
    lines.append(
        f"    [{lo:.1f}, {lo + 0.2:.1f})   "
        f"{np.average(p_val[m], weights=n[m]):.3f} -> "
        f"{wins[m].sum() / n[m].sum():.3f}   ({int(n[m].sum())})"
    )
  good = has & (n >= 20)
  if good.sum() > 2:
    r_or = np.corrcoef(po[good], p_net[good])[0, 1]
    r_val = np.corrcoef(p_val[good], p_net[good])[0, 1]
    lines.append(
        f"  correlation with the network's success, over {int(good.sum())} "
        f"class-seats with 20+ deals: oracle {r_or:.2f}, value head {r_val:.2f}"
    )
  return lines


def rows(res: dict):
  """The CSV rows of one contract (module doc)."""
  name = res["contract"]
  marked = sf.trump_marked_of(sf.CONTRACTS[name])
  points = float(res["points"])
  counts = sf.class_counts(marked)
  out = []
  for i, key in enumerate(res["classes"]):
    for seat in range(sf.NUM_PLAYERS):
      n = res["n"][i][seat]
      if not n:
        continue
      p_net = res["wins"][i][seat] / n
      p_val = min(max((res["value_sum"][i][seat] / n / points + 1) / 2, 0), 1)
      out.append((
          name,
          hand_text(key, marked),
          seat,
          int(counts[i]),
          n,
          round(p_net, 3),
          round(p_val, 3),
          round(res["p_oracle"][i][seat], 3),
      ))
  out.sort(key=lambda r: (-r[5], r[1], r[2]))
  return out


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("measurements", nargs="+")
  p.add_argument("--csv", default="")
  a = p.parse_args(argv)
  table = []
  for path in a.measurements:
    with open(path) as f:
      res = json.load(f)
    print("\n".join(summary(res)))
    table += rows(res)
  if a.csv:
    with open(a.csv, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow([
          "contract",
          "hand",
          "declarer_seat",
          "hands_dealt",
          "deals_played",
          "p_network",
          "p_value_head",
          "p_oracle",
      ])
      w.writerows(table)
    print(f"wrote {len(table)} rows to {a.csv}")


if __name__ == "__main__":
  main()
