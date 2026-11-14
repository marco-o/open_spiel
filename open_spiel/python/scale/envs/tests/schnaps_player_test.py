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

"""envs.schnaps_player: routing, the artifact and the build script."""

import sys

import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as G
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.configs import schnaps_forced_cpu as forced
from open_spiel.python.scale.configs import schnaps_normal_cpu as normal
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.envs import schnaps_bidding as bidding
from open_spiel.python.scale.envs import schnaps_elena as elena
from open_spiel.python.scale.envs import schnaps_player as player_lib
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.examples import schnaps_build_player as build_script
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.policies import registry

# pylint: enable=g-import-not-at-top,wrong-import-position


@pytest.fixture(scope="module", autouse=True)
def _x64():
  x64.enable_x64()


def _part(name, seed):
  kwargs = (
      normal.policy_kwargs() if name == "normal" else forced.policy_kwargs(name)
  )
  return refs.PolicyRef(arch_name="network", arch_kwargs=kwargs).build(
      jax.random.key(seed)
  )


@pytest.fixture(scope="module")
def parts():
  return {n: _part(n, i) for i, n in enumerate(player_lib.PART_NAMES)}


@pytest.fixture(scope="module")
def player(parts):
  policy = player_lib.SchnapsPlayerPolicy(
      parts={n: p for n, (p, _) in parts.items()}, rule="greedy"
  )
  params = {n: q for n, (_, q) in parts.items()}
  return policy, params


def test_the_registry_rebuilds_the_player(player):
  policy, _ = player
  again = registry.build(player_lib.ARCH, policy.spec.arch_kwargs)
  assert again.spec == policy.spec
  assert policy.spec.game == "schnaps"
  assert policy.spec.arch_kwargs["rule"] == "greedy"


def _snapshots(n_games=30):
  """(info, legal, seat) at every decision of random schnaps games."""
  env = schnaps.SchnapsEnv()
  step, info_state = jax.jit(env.step), jax.jit(env.info_state)
  legal_mask = jax.jit(env.legal_mask)
  rng = np.random.default_rng(0)
  rows = []
  for g in range(n_games):
    s = env.init(jax.random.key(g))
    while not bool(s.is_terminal):
      seat = int(s.current_player)
      legal = np.asarray(legal_mask(s))
      rows.append((info_state(s, seat), legal, seat))
      a = rng.choice(np.flatnonzero(legal))
      if int(s.phase) == G.PHASE_BIDDING and rng.random() < 0.7:
        a = G.PASS_ACTION  # mostly pass, so some deals stay Normal
      s = step(s, jnp.int32(a), jax.random.key(g))
  info = jax.tree.map(lambda *x: jnp.stack(x), *[r[0] for r in rows])
  legal = jnp.asarray(np.stack([r[1] for r in rows]))
  seat = jnp.asarray([r[2] for r in rows], jnp.int32)
  return info, legal, seat


def _step(info, legal, seat):
  b = seat.shape[0]
  return types.SeatStep(
      info=info,
      legal_mask=legal,
      seat=seat,
      to_move=seat,
      new_hand=jnp.zeros((b,), jnp.bool_),
      new_match=jnp.zeros((b,), jnp.bool_),
      last_action=jnp.full((b,), types.NO_ACTION, jnp.int32),
      last_actor=jnp.full((b,), types.NO_ACTOR, jnp.int32),
  )


def test_routing_by_phase_and_contract(player, parts):
  policy, params = player
  info, legal, seat = _snapshots()
  step = _step(info, legal, seat)
  key = jax.random.key(0)
  b = seat.shape[0]
  carry = policy.init_carry(params, b, seat, key)
  out = jax.jit(policy.evaluate)(params, carry, step, key)
  phase = np.asarray(info["phase"])
  contract = np.asarray(info["contract"])
  part_out = {}
  for n, (p, q) in parts.items():
    part_out[n] = p.evaluate(q, p.init_carry(q, b, seat, key), step, key)
  logits = np.asarray(out.logits)
  trump = phase == G.PHASE_CHOOSE_TRUMP
  assert trump.any()
  np.testing.assert_allclose(
      logits[trump], np.asarray(part_out["normal"].logits)[trump], rtol=1e-5, atol=1e-5
  )
  for c, n in enumerate(player_lib.PART_NAMES):
    rows = (phase == G.PHASE_PLAY) & (contract == c)
    assert rows.any(), n
    np.testing.assert_allclose(
        logits[rows], np.asarray(part_out[n].logits)[rows], rtol=1e-5, atol=1e-5
    )
  bid = phase == G.PHASE_BIDDING
  assert bid.any()
  chosen = logits[bid].argmax(axis=1)
  assert np.asarray(legal)[bid][np.arange(bid.sum()), chosen].all()
  assert (chosen >= G.PASS_ACTION).all()
  # The bid is the greedy choice on the estimates in aux.
  bv = np.asarray(out.aux["bid_value"])[bid]
  assert (chosen == bv.argmax(axis=1)).all()
  assert np.isneginf(np.asarray(out.aux["bid_value"])[~bid]).all()


def _write_sources(tmp_path, parts):
  sources = {}
  for n, (p, q) in parts.items():
    env_name = "schnaps_normal" if n == "normal" else forced.env_name(n)
    env = env_registry.make(env_name)
    d = str(tmp_path / n)
    artifact.write_artifact(
        d,
        7,
        p,
        q,
        rules_sha=env.declared_rules_sha,
        env_spec=env_registry.spec_of(env).to_json(),
        env_name=env.name,
        extensions={"value_scale": 3.0},
    )
    sources[n] = f"{d}@7"
  return sources


def test_build_script_and_a_match(tmp_path, parts):
  sources = _write_sources(tmp_path, parts)
  out = str(tmp_path / "player")
  build_script.build(sources, out, 0, "greedy", {"margin": 0.5})
  with pytest.raises(ValueError):
    build_script.build(sources, out, 0, "greedy", {})
  build_script.build(sources, out, 1, "pass", {})
  art = artifact.load_artifact(out, 0)
  assert art.manifest.game == "schnaps"
  assert art.manifest.extensions["value_scale"] == 3.0
  assert len(art.manifest.parents) == 4
  assert art.policy.rule.margin == 0.5
  for n, (_, q) in parts.items():
    for a, b in zip(
        jax.tree_util.tree_leaves(art.params[n]), jax.tree_util.tree_leaves(q)
    ):
      assert np.array_equal(np.asarray(a), np.asarray(b))
  # It plays the full game against Elena in the evaluator.
  env = env_registry.training_env(art.manifest)
  ev = elena.evaluator(env)
  subject = match.Entrant.from_artifact(art, rule=types.ARGMAX, name="p")
  res = ev.head_to_head(subject, elena.entrant(), 4, jax.random.key(0))
  assert np.isfinite(res.mean.mean)
  # The pass baseline never bids.
  base = artifact.load_artifact(out, 1)
  assert isinstance(base.policy.rule, bidding.PassRule)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
