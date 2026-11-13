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

"""Tests of envs.schnaps_oracle: its rules against pyspiel, its solver."""

import random
import sys

import numpy as np
import pytest

import pyspiel
from open_spiel.python.scale.envs import schnaps_oracle as oracle

GAME = "schnaps(max_flecks=2)"


def _random_bid(rng, legal, p_pass=0.7):
  """Mostly passes, so most deals reach the card play."""
  if oracle.PASS in legal and rng.random() < p_pass:
    return oracle.PASS
  return rng.choice(legal)


def _play_both(seed):
  """A random deal in pyspiel and in the Model, checked at every step."""
  rng = random.Random(seed)
  game = pyspiel.load_game(GAME)
  s = game.new_initial_state()
  m = oracle.Model(max_flecks=2)
  while not s.is_terminal():
    assert not m.is_terminal()
    assert s.is_chance_node() == m.is_chance()
    if s.is_chance_node():
      a = rng.choice([a for a, _ in s.chance_outcomes()])
    else:
      assert s.current_player() == m.current_player()
      legal = s.legal_actions()
      assert sorted(legal) == sorted(m.legal_actions()), str(s)
      a = _random_bid(rng, legal) if m.phase == oracle.BIDDING else (
          rng.choice(legal))
    s.apply_action(a)
    m.apply(a)
  assert m.is_terminal()
  assert list(s.returns()) == m.returns
  return m


@pytest.mark.parametrize("seed", range(300))
def test_model_matches_pyspiel(seed):
  _play_both(seed)


def test_model_reaches_every_contract():
  seen = {_play_both(seed).contract for seed in range(300)}
  assert {oracle.NORMAL, oracle.SCHNAPSER, oracle.LOMT, oracle.BAUERN} <= seen


def _pyspiel_minimax(s):
  """Plain minimax on pyspiel, value for team 0, every seat perfect."""
  if s.is_terminal():
    return s.returns()[0]
  vals = []
  for a in s.legal_actions():
    c = s.clone()
    c.apply_action(a)
    vals.append(_pyspiel_minimax(c))
  return max(vals) if s.current_player() % 2 == 0 else min(vals)


def _midgame(seed, plays_left):
  """pyspiel and Model states with `plays_left` card plays or fewer to go."""
  rng = random.Random(seed)
  game = pyspiel.load_game(GAME)
  s = game.new_initial_state()
  m = oracle.Model(max_flecks=2)
  while not s.is_terminal():
    if m.phase == oracle.PLAY and 20 - len(m.plays) <= plays_left:
      return s, m
    if s.is_chance_node():
      a = rng.choice([a for a, _ in s.chance_outcomes()])
    elif m.phase == oracle.BIDDING:
      a = _random_bid(rng, s.legal_actions(), p_pass=0.95)
    else:
      a = rng.choice(s.legal_actions())
    s.apply_action(a)
    m.apply(a)
  return None, None


@pytest.mark.parametrize("seed", range(40))
def test_play_solver_matches_pyspiel_minimax(seed):
  s, m = _midgame(seed, plays_left=11)
  if s is None:
    pytest.skip("the deal ended early")
  o = oracle.Oracle(m.hands(), m.trump)
  vals = o.action_values(m)
  for a, v in vals.items():
    c = s.clone()
    c.apply_action(a)
    assert v == _pyspiel_minimax(c), (a, str(s))


@pytest.mark.parametrize("seed", range(10))
def test_full_deal_solves(seed):
  s, m = _midgame(seed, plays_left=20)
  if m is None:
    pytest.skip("the deal ended before the play")
  o = oracle.Oracle(m.hands(), m.trump)
  v = o.solver(m.rules()).value(m.play)
  assert -24 <= v <= 24
  assert o.decide(m) in m.legal_actions()


def test_bid_value_is_minimax_over_the_bidding():
  rng = random.Random(7)
  m = oracle.Model(max_flecks=2)
  while m.phase != oracle.BIDDING:
    if m.is_chance():
      m.apply(rng.choice(m.legal_actions()))
    else:
      m.apply(oracle.FIRST_TRUMP)
  o = oracle.Oracle(m.hands(), m.trump)
  vals = o.action_values(m)
  assert set(vals) == set(m.legal_actions())
  assert o.bid_value(m.bid_state()) == max(vals.values())  # seat 0 bids
  assert o.decide(m) in vals


@pytest.mark.parametrize("seed", range(3))
def test_info_state_matches_the_jax_env(seed):
  import jax  # pylint: disable=g-import-not-at-top
  from open_spiel.python.scale.envs import schnaps  # pylint: disable=g-import-not-at-top
  env = schnaps.SchnapsEnv()
  g = env.game
  rng = random.Random(seed)
  m = oracle.Model(max_flecks=2)
  s = g.new_initial_state()
  while not m.is_terminal():
    if not m.is_chance():
      for seat in range(4):
        want = jax.tree.map(np.asarray, env.info_state(s, seat))
        got = m.info_state(seat)
        assert set(want) == set(got)
        for k in want:
          np.testing.assert_array_equal(got[k], want[k], err_msg=k)
          assert got[k].dtype == want[k].dtype, k
    a = rng.choice(m.legal_actions())
    m.apply(a)
    s = g.apply_action(s, jax.numpy.asarray(a, jax.numpy.int32))


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-q"]))
