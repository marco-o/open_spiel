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

"""Tests of envs.schnaps_search: the deal sampler and the search bot."""

import random
import sys

import pytest

from open_spiel.python.scale.envs import schnaps_oracle as oracle
from open_spiel.python.scale.envs import schnaps_search as search


def _deal_to_play(rng, model=None):
  """A Model in the card play, mostly passing in the bidding."""
  m = model or oracle.Model()
  while m.phase != oracle.PLAY and not m.is_terminal():
    legal = m.legal_actions()
    if m.phase == oracle.BIDDING and oracle.PASS in legal and rng.random() < .9:
      m.apply(oracle.PASS)
    else:
      m.apply(rng.choice(legal))
  return m


def _action_of(card, declared):
  if not declared:
    return card
  s, r = oracle.suit_of(card), oracle.rank_of(card)
  return oracle.FIRST_DECLARE + 2 * s + (0 if r == oracle.KOENIG else 1)


def _consistent(model, hands):
  """Whether every past play was legal had `hands` been the hidden hands."""
  start = list(hands)
  for c, p in zip(model.plays, model.play_seat):
    start[p] |= 1 << c
  if any(bin(h).count("1") != oracle.HAND_SIZE for h in start):
    return False
  rules = model.rules()
  p = oracle.initial_play(tuple(start), rules)
  for c, d in zip(model.plays, model.declared):
    a = _action_of(c, d)
    if a not in oracle.play_legal(rules, p):
      return False
    p, _ = oracle.play_apply(rules, p, a)
  return True


def _walk(seed, check):
  """Plays a random deal, calling check(model, seat) before each play."""
  rng = random.Random(seed)
  m = _deal_to_play(rng)
  while not m.is_terminal():
    check(m, m.current_player(), rng)
    m.apply(rng.choice(m.legal_actions()))


@pytest.mark.parametrize("seed", range(150))
def test_the_true_deal_is_never_ruled_out(seed):
  def check(m, seat, rng):
    del rng
    known, allowed, need = search.constraints(m, seat)
    truth = m.hands()
    for c, p in known.items():
      assert truth[p] >> c & 1, (c, p)
    for c, seats in allowed.items():
      holder = next(q for q in range(4) if truth[q] >> c & 1)
      assert holder in seats, (c, holder, seats)
    for q in range(4):
      if q != seat:
        held = bin(truth[q]).count("1")
        assert need[q] == held - sum(1 for p in known.values() if p == q)
  _walk(seed, check)


@pytest.mark.parametrize("seed", range(150))
def test_samples_are_consistent_with_the_public_play(seed):
  def check(m, seat, rng):
    for _ in range(3):
      hands = search.sample_hands(m, seat, rng)
      assert hands is not None
      assert hands[seat] == m.hands()[seat]
      union = 0
      for h in hands:
        assert union & h == 0
        union |= h
      assert union == sum(m.hands())  # the same cards are still out
      assert _consistent(m, hands), (m.plays, hands)
  _walk(seed, check)


def test_samples_vary():
  rng = random.Random(0)
  m = _deal_to_play(rng)
  draws = {search.sample_hands(m, 0, rng) for _ in range(20)}
  assert len(draws) > 10


@pytest.mark.parametrize("seed", range(20))
def test_bot_plays_legal_moves(seed):
  bot = search.SearchBot(n_samples=4, seed=seed,
                         phases=(oracle.BIDDING, oracle.PLAY))
  rng = random.Random(seed)
  m = oracle.Model()
  while not m.is_terminal():
    a = None if m.is_chance() else bot.decide(m)
    if a is None:
      a = rng.choice(m.legal_actions())
    assert a in m.legal_actions()
    m.apply(a)
  assert bot.failed_samples == 0


def test_bot_does_not_search_outside_its_phases():
  rng = random.Random(1)
  m = oracle.Model()
  while m.phase != oracle.BIDDING:
    m.apply(rng.choice(m.legal_actions()))
  assert search.SearchBot(n_samples=2).decide(m) is None


@pytest.mark.parametrize("seed", range(10))
def test_bot_is_exact_when_nothing_is_hidden(seed):
  """With two cards left in play, nothing is hidden from the seat to move."""
  rng = random.Random(seed)
  m = _deal_to_play(rng)
  while not m.is_terminal() and len(m.plays) < 18:
    m.apply(rng.choice(m.legal_actions()))
  if m.is_terminal():
    pytest.skip("ended early")
  bot = search.SearchBot(n_samples=3)
  vals = bot.action_values(m)
  exact = oracle.Oracle(m.hands(), m.trump).action_values(m)
  assert vals == pytest.approx(exact)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-q"]))
