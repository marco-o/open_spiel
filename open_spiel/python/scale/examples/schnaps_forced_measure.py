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

r"""Measures a forced-contract network's real success rate per hand class.

  python -m open_spiel.python.scale.examples.schnaps_forced_measure \
      --contract=lomt --deals_per_cell=300

Round 2 of the forced-contract plan. The network at --checkpoint plays all
four seats greedily (its most likely action) on forced deals of its env
(envs.schnaps_forced). The declarer's hand is drawn uniformly over the
classes the hands file keeps at that seat (weight > 0), not by the training
weights, so every kept class gets about --deals_per_cell deals per seat.
Classes the file drops (the oracle almost never wins them) are not measured.

Per (class, declarer seat) it records:
  * n: deals played;
  * wins: deals the declarer's side won (the contract made);
  * value: the sum over deals of the network's value head at the first
    play, read from the declarer's own view, in game points for the
    declarer's side. For a contract worth v that is won or lost,
    p = (E / v + 1) / 2 turns the mean E into a predicted chance of success.

Output: a JSON file (--out) with the classes, the oracle's p and the counts
above; examples/schnaps_forced_report.py turns it into tables.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import schnaps_forced as sf

SCHEMA = "schnaps-forced-measure/1"


def uniform_hands_file(contract_name: str, path: str) -> np.ndarray:
  """Writes a hands file uniform over the kept classes of each seat.

  Args:
    contract_name: the contract.
    path: where to write it.

  Returns:
    bool[num_classes, 4], whether each class is kept at each seat.
  """
  data, _ = sf.load_hands()
  entry = data["contracts"][contract_name]
  marked = sf.trump_marked_of(sf.CONTRACTS[contract_name])
  counts = sf.class_counts(marked).astype(np.float64)
  w = np.asarray(entry["weight"], np.float64)
  if w.ndim == 1:
    w = np.repeat(w[:, None], sf.NUM_PLAYERS, axis=1)
  kept = w > 0
  uniform = np.where(kept, 1.0 / counts[:, None], 0.0)
  out = {
      "schema": sf.HANDS_SCHEMA,
      "contracts": {
          contract_name: {
              "classes": entry["classes"],
              "weight": uniform.tolist(),
          }
      },
  }
  with open(path, "w") as f:
    json.dump(out, f)
  return kept


def play_batch(env, policy, params, key, batch: int):
  """Deals `batch` forced deals and plays them out with `policy` everywhere.

  Args:
    env: the forced env.
    policy: the network policy (stateless).
    params: its params.
    key: the PRNG key.
    batch: the number of deals.

  Returns:
    (first states, terminal states, the declarer's value at the first play
    in the declarer side's game points).
  """
  k_deal, k_eval = jax.random.split(key)
  s0 = jax.vmap(env.init)(jax.random.split(k_deal, batch))
  value_scale = 3.0

  def evaluate(s, seat, legal):
    info = jax.vmap(env.info_state)(s, seat)
    step = types.SeatStep(
        info=info,
        legal_mask=legal,
        seat=seat,
        to_move=seat,
        new_hand=jnp.zeros((batch,), jnp.bool_),
        new_match=jnp.zeros((batch,), jnp.bool_),
        last_action=jnp.full((batch,), types.NO_ACTION, jnp.int32),
        last_actor=jnp.full((batch,), types.NO_ACTOR, jnp.int32),
    )
    carry = policy.init_carry(params, batch, seat, k_eval)
    return policy.evaluate(params, carry, step, k_eval)

  decl = s0.declarer.astype(jnp.int32)
  legal0 = jax.vmap(env.legal_mask)(s0)
  out = evaluate(s0, decl, legal0)
  v = jnp.take_along_axis(out.value, decl[:, None], axis=1)[:, 0] * value_scale

  def body(_, s):
    actor = jnp.maximum(jax.vmap(env.actor)(s), 0)
    legal = jax.vmap(env.legal_mask)(s)
    logits = evaluate(s, actor, legal).logits
    a = jnp.argmax(jnp.where(legal, logits, -jnp.inf), axis=-1).astype(jnp.int32)
    nxt = jax.vmap(env.step)(s, a, jax.random.split(k_eval, batch))
    done = jax.vmap(env.terminal)(s)
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            done.reshape(done.shape + (1,) * (x.ndim - 1)), x, y
        ),
        s,
        nxt,
    )

  s1 = jax.lax.fori_loop(0, env.max_len, body, s0)
  return s0, s1, v


def class_ids(s0, marked: bool) -> np.ndarray:
  """Each deal's declarer-hand class index."""
  holder = np.asarray(s0.holder)
  d = np.asarray(s0.declarer)
  t = np.asarray(s0.trump)
  masks = sf.suit_masks(holder == d[:, None])
  classes, _ = sf.hand_classes(marked)
  index = {k: i for i, k in enumerate(classes)}
  return np.array(
      [index[sf.class_key(m, int(tt), marked)] for m, tt in zip(masks, t)]
  )


def measure(contract_name, checkpoint, deals_per_cell, batch, seed):
  """Runs the measurement (module doc).

  Args:
    contract_name: the contract.
    checkpoint: "<dir>@<step>".
    deals_per_cell: deals per kept (class, seat), on average.
    batch: deals per jitted batch.
    seed: the seed.

  Returns:
    The JSON-able result.
  """
  contract = sf.CONTRACTS[contract_name]
  marked = sf.trump_marked_of(contract)
  path, step = checkpoint.rsplit("@", 1)
  art = artifact.load_artifact(os.path.expanduser(path), int(step))
  policy, params = art.policy, art.params
  if policy.spec.stateful:
    sys.exit("the measurement plays stateless policies only")
  with tempfile.TemporaryDirectory() as tmp:
    hands = os.path.join(tmp, "uniform.json")
    kept = uniform_hands_file(contract_name, hands)
    env = sf.CONTRACT_ENVS[contract_name](hands_path=hands)
  if art.manifest.game != env.name:
    sys.exit(f"{checkpoint} plays {art.manifest.game}, not {env.name}")
  cells = int(kept.sum())
  total = deals_per_cell * cells
  n_batches = -(-total // batch)
  n = np.zeros(kept.shape, np.int64)
  wins = np.zeros(kept.shape, np.int64)
  value = np.zeros(kept.shape, np.float64)
  run = jax.jit(lambda k: play_batch(env, policy, params, k, batch))
  key = jax.random.key(seed)
  t0 = time.time()
  for b in range(n_batches):
    s0, s1, v = run(jax.random.fold_in(key, b))
    ids = class_ids(s0, marked)
    d = np.asarray(s0.declarer)
    r = np.asarray(jax.vmap(env.returns)(s1))[np.arange(batch), d]
    assert bool(np.asarray(jax.vmap(env.terminal)(s1)).all())
    np.add.at(n, (ids, d), 1)
    np.add.at(wins, (ids, d), (r > 0).astype(np.int64))
    np.add.at(value, (ids, d), np.asarray(v, np.float64))
    if b % 20 == 0 or b == n_batches - 1:
      done = (b + 1) * batch
      el = time.time() - t0
      print(
          f"{contract_name}: {done}/{n_batches * batch} deals, "
          f"{done / max(el, 1e-9):.0f}/s, success so far "
          f"{wins.sum() / max(n.sum(), 1):.3f}",
          flush=True,
      )
  data, hands_sha = sf.load_hands()
  entry = data["contracts"][contract_name]
  p_oracle = np.asarray(entry["p"], np.float64)
  if p_oracle.ndim == 1:
    p_oracle = np.repeat(p_oracle[:, None], sf.NUM_PLAYERS, axis=1)
  classes, _ = sf.hand_classes(marked)
  return {
      "schema": SCHEMA,
      "contract": contract_name,
      "points": int(sf.schnaps.CONTRACT_POINTS[contract]),
      "checkpoint": checkpoint,
      "hands_sha256": hands_sha,
      "deals_per_cell": deals_per_cell,
      "seed": seed,
      "classes": [list(k) for k in classes],
      "kept": kept.tolist(),
      "p_oracle": p_oracle.round(4).tolist(),
      "n": n.tolist(),
      "wins": wins.tolist(),
      "value_sum": value.round(4).tolist(),
  }


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  p.add_argument("--contract", required=True, choices=sorted(sf.CONTRACTS))
  p.add_argument("--checkpoint", default="", help="default ~/schnaps_forced_<c>@1000")
  p.add_argument("--deals_per_cell", type=int, default=300)
  p.add_argument("--batch", type=int, default=1024)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="", help="default ~/schnaps_forced_measure_<c>.json")
  a = p.parse_args(argv)
  x64.enable_x64()
  ckpt = a.checkpoint or f"~/schnaps_forced_{a.contract}@1000"
  out = os.path.expanduser(a.out or f"~/schnaps_forced_measure_{a.contract}.json")
  res = measure(a.contract, ckpt, a.deals_per_cell, a.batch, a.seed)
  with open(out, "w") as f:
    json.dump(res, f, separators=(",", ":"))
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
