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

"""envs.schnaps_bidding: counterfactual bids, estimates and bid rules."""

import sys

import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as G
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.envs import schnaps_bidding as bidding

# pylint: enable=g-import-not-at-top,wrong-import-position

BID_KEYS = (
    "phase", "contract", "declarer", "flecks", "passes", "any_call",
    "last_actor", "bids", "n_bids",
)


@pytest.fixture(scope="module", autouse=True)
def _x64():
  x64.enable_x64()


@pytest.fixture(scope="module")
def env():
  return schnaps.SchnapsEnv()


def _jitted(env):
  return (
      jax.jit(env.step),
      jax.jit(env.info_state),
      jax.jit(env.legal_mask),
      jax.jit(bidding.counterfactual),
  )


def _to_bidding(env, seed):
  """A random deal advanced to its first bid."""
  step, _, legal_mask, _ = _jitted(env)
  rng = np.random.default_rng(seed)
  s = env.init(jax.random.key(seed))
  while int(s.phase) != G.PHASE_BIDDING:
    a = rng.choice(np.flatnonzero(np.asarray(legal_mask(s))))
    s = step(s, jnp.int32(a), jax.random.key(seed + 1000))
  return s, rng


def _batch(info):
  return jax.tree.map(lambda x: jnp.asarray(x)[None], info)


def test_counterfactual_equals_the_game(env):
  """Bid, then passes to the end: the game's InfoState, for every legal bid."""
  step, info_state, legal_mask, cf = _jitted(env)
  key = jax.random.key(7)
  checked = set()
  for seed in range(25):
    s, rng = _to_bidding(env, seed)
    while int(s.phase) == G.PHASE_BIDDING:
      seat = int(s.current_player)
      info = info_state(s, seat)
      legal = np.flatnonzero(np.asarray(legal_mask(s)))
      for a in legal:
        got = cf(info, jnp.int32(a))
        real = step(s, jnp.int32(a), key)
        while int(real.phase) == G.PHASE_BIDDING:
          real = step(real, jnp.int32(G.PASS_ACTION), key)
        want = info_state(real, seat)
        keys = BID_KEYS if int(real.phase) == G.PHASE_OVER else want.keys()
        for k in keys:
          np.testing.assert_array_equal(
              np.asarray(got[k]), np.asarray(want[k]), err_msg=k
          )
          assert got[k].dtype == want[k].dtype, k
        checked.add(int(a))
      s = step(s, jnp.int32(rng.choice(legal)), key)
  # Pass, fleck and at least the first calls were exercised.
  assert {G.PASS_ACTION, G.FLECK_ACTION, 34, 35, 36} <= checked


def _const_value_fn(table):
  table = jnp.asarray(table, jnp.float32)

  def value_fn(info, contract):
    b = info["seat"].shape[0]
    if isinstance(contract, int):
      return jnp.full((b,), table[contract])
    return table[contract]

  return value_fn


def _first_bid(env, seed=0):
  s, _ = _to_bidding(env, seed)
  return s, _batch(env.info_state(s, int(s.current_player)))


def _legal(*actions):
  m = np.zeros((1, bidding.NUM_ACTIONS), bool)
  m[0, list(actions)] = True
  return jnp.asarray(m)


def _ctx(info, legal, est):
  return bidding.BidContext(info, legal, est, bidding.set_context(info))


def test_estimates_and_the_greedy_rule(env):
  _, info = _first_bid(env)
  calls = [bidding.call_action(c) for c in range(1, 6)]
  legal = _legal(G.PASS_ACTION, *calls)
  # Normal +1, Schnapser -2, Lomt +1.8, Bauern -6 (points for my pair).
  est = bidding.estimate(info, legal, _const_value_fn([1.0, -2.0, 1.8, -6.0]))
  v = np.asarray(est.value[0])
  assert v[G.PASS_ACTION] == 1.0 and v[34] == -2.0 and v[35] == 1.8
  assert v[37] == 18.0 and v[38] == 24.0
  assert v[G.FLECK_ACTION] == -np.inf  # not legal here
  assert np.isclose(est.p_success[0, 35], (1.8 / 9 + 1) / 2)
  assert np.isnan(est.p_success[0, G.PASS_ACTION])  # Normal: no stake
  assert est.stake[0, 36] == 12.0
  greedy = bidding.make_rule("greedy")
  # Hermort wins at once: always the best.
  assert int(greedy.choose(_ctx(info, legal, est), None)[0]) == 38
  legal = _legal(G.PASS_ACTION, *calls[:3])
  est = bidding.estimate(info, legal, _const_value_fn([1.0, -2.0, 1.8, -6.0]))
  assert int(greedy.choose(_ctx(info, legal, est), None)[0]) == 35  # Lomt
  cautious = bidding.make_rule("greedy", {"margin": 1.0})
  assert int(cautious.choose(_ctx(info, legal, est), None)[0]) == G.PASS_ACTION
  never = bidding.make_rule("pass")
  assert int(never.choose(_ctx(info, legal, est), None)[0]) == G.PASS_ACTION


def test_pass_and_fleck_against_a_call(env):
  """After an opponent's Bauern: pass and fleck read the Bauern value."""
  step, info_state, legal_mask, _ = _jitted(env)
  s, _ = _to_bidding(env, 3)
  key = jax.random.key(0)
  # Seat 0 passes, seat 1 calls Bauern; seat 2 (seat 0's partner) to bid.
  s = step(s, jnp.int32(G.PASS_ACTION), key)
  s = step(s, jnp.int32(bidding.call_action(G.BAUERN)), key)
  assert int(s.current_player) == 2
  info = _batch(info_state(s, 2))
  legal = jnp.asarray(legal_mask(s))[None]
  legal = legal & (jnp.arange(bidding.NUM_ACTIONS) < 37)  # no Jodler/Hermort
  assert bool(legal[0, G.FLECK_ACTION])
  # From seat 2's view the Bauern is worth +4 (the declarer likely fails).
  est = bidding.estimate(info, legal, _const_value_fn([0.0, 0.0, 0.0, 4.0]))
  assert float(est.value[0, G.PASS_ACTION]) == 4.0
  assert float(est.value[0, G.FLECK_ACTION]) == 8.0
  assert float(est.stake[0, G.FLECK_ACTION]) == 24.0
  assert not bool(est.my_side_declares[0, G.PASS_ACTION])
  # The declarer's chance: (-4 / 12 + 1) / 2.
  assert np.isclose(float(est.p_success[0, G.PASS_ACTION]), 1 / 3)
  greedy = bidding.make_rule("greedy")
  ctx = _ctx(info, legal, est)
  assert int(greedy.choose(ctx, None)[0]) == G.FLECK_ACTION


class _NeedRule:
  """A test rule that reads the set context: calls only when far behind."""

  name = "test_need"

  def kwargs(self):
    return {}

  def choose(self, ctx, key):
    del key
    behind = ctx.set.known & (ctx.set.our_need > ctx.set.their_need + 10)
    lomt = bidding.call_action(G.LOMT)
    can = ctx.legal[:, lomt]
    return jnp.where(behind & can, lomt, bidding.PASS).astype(jnp.int32)


def test_a_context_rule_plugs_in(env):
  bidding.register_rule("test_need", _NeedRule)
  with pytest.raises(ValueError):
    bidding.register_rule("test_need", _NeedRule)
  _, info = _first_bid(env)
  legal = _legal(G.PASS_ACTION, 34, 35)
  est = bidding.estimate(info, legal, _const_value_fn([0.0] * 4))
  rule = bidding.make_rule("test_need")
  assert not bool(bidding.set_context(info).known[0])
  assert int(rule.choose(_ctx(info, legal, est), None)[0]) == G.PASS_ACTION
  seat = int(info["seat"][0])
  need = np.zeros((1, 2), np.int32)
  need[0, seat % 2], need[0, 1 - seat % 2] = 25, 3
  info = dict(info, set_need=jnp.asarray(need))
  ctx = _ctx(info, legal, est)
  assert int(ctx.set.our_need[0]) == 25 and int(ctx.set.their_need[0]) == 3
  assert int(rule.choose(ctx, None)[0]) == bidding.call_action(G.LOMT)
  with pytest.raises(KeyError):
    bidding.make_rule("no_such_rule")


def _after_opponent_bauern(env):
  """Seat 2 to bid after seat 1's Bauern: (info [1], legal [1])."""
  step, info_state, legal_mask, _ = _jitted(env)
  s, _ = _to_bidding(env, 3)
  key = jax.random.key(0)
  s = step(s, jnp.int32(G.PASS_ACTION), key)
  s = step(s, jnp.int32(bidding.call_action(G.BAUERN)), key)
  info = _batch(info_state(s, 2))
  legal = jnp.asarray(legal_mask(s))[None]
  legal = legal & (jnp.arange(bidding.NUM_ACTIONS) < 37)  # no Jodler/Hermort
  return info, legal


def _with_need(info, our, their):
  """Seat 2's info with the set situation (seat 2 is partnership 0)."""
  return dict(info, set_need=jnp.asarray([[our, their]], jnp.int32))


def _set_aware(info, legal, bauern_value):
  est = bidding.estimate(
      info, legal, _const_value_fn([0.0, 0.0, 0.0, bauern_value])
  )
  rule = bidding.make_rule("set_aware")
  return int(rule.choose(_ctx(info, legal, est), None)[0])


def test_set_aware_flecks_only_when_sure_without_a_set(env):
  info, legal = _after_opponent_bauern(env)
  # +4 for us: the declarer's chance 1/3, ours 2/3 < 0.75: no fleck.
  assert _set_aware(info, legal, 4.0) == G.PASS_ACTION
  # +8: ours 5/6: fleck.
  assert _set_aware(info, legal, 8.0) == G.FLECK_ACTION


def test_set_aware_flecks_when_it_cannot_cost_the_set(env):
  info, legal = _after_opponent_bauern(env)
  # They need 9 <= 12: a Bauern won ends the set for them anyway, so the
  # fleck is free even though we are only 7/12 sure.
  assert _set_aware(_with_need(info, 31, 9), legal, 2.0) == G.FLECK_ACTION
  # Far from the end on both sides, the same estimate does not fleck.
  assert _set_aware(_with_need(info, 31, 31), legal, 2.0) == G.PASS_ACTION


def test_set_aware_does_not_fleck_close_to_the_end(env):
  info, legal = _after_opponent_bauern(env)
  # We need 3: winning more is worthless, losing 24 instead of 12 is not.
  assert _set_aware(_with_need(info, 3, 20), legal, 8.0) == G.PASS_ACTION
  rule = bidding.make_rule("set_aware", {"fleck_sure": 0.9})
  assert rule.kwargs()["fleck_sure"] == 0.9


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
