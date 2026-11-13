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

"""envs.schnaps_elena: each rule on set deals, legality, and her strength."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import schnaps as game
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.envs import schnaps_elena as elena
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import registry as eval_registry
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform

ENV = schnaps.SchnapsEnv()
G = ENV.game
_APPLY = jax.jit(G.apply_action)
_LEGAL = jax.jit(G.legal_actions_mask)
_RULE = jax.jit(elena.rule_action)
_INFO = jax.jit(ENV.info_state)

HERZ, CARO, PIK, KREUZ = range(4)
U, O, K, Z, A = range(5)


def c(suit: int, rank: int) -> int:
  """The card id of (suit, rank).

  Args:
    suit: 0..3.
    rank: 0..4 (Unter .. As).

  Returns:
    The card.
  """
  return suit * 5 + rank


def _fill(hands):
  """The four hands, each completed to five cards from the unused ones.

  Args:
    hands: up to four lists of cards (seat 0 first); missing cards are taken
      in order from the cards no hand names.

  Returns:
    Four lists of five cards.
  """
  hands = [list(h) for h in hands] + [[] for _ in range(4 - len(hands))]
  free = [x for x in range(20) if all(x not in h for h in hands)]
  for h in hands:
    while len(h) < 5:
      h.append(free.pop(0))
  return hands


def _deal(hands, trump, bids=()):
  """The state after dealing `hands`, choosing `trump` and the `bids`.

  Args:
    hands: the hands (_fill); seat 0's first three cards are its first three.
    trump: the trump suit.
    bids: bid actions applied in turn from seat 0.

  Returns:
    The state.
  """
  hands = _fill(hands)
  s = G.new_initial_state()
  order = (
      hands[0][:3]
      + [game.FIRST_TRUMP_ACTION + trump]
      + hands[0][3:]
      + hands[1]
      + hands[2]
      + hands[3]
  )
  for a in list(order) + list(bids):
    s = _APPLY(s, jnp.int32(a))
  return s


def _elena(s, key=0):
  """Elena's rule action for the seat on move.

  Args:
    s: the state.
    key: the PRNG seed.

  Returns:
    (action, legal mask).
  """
  p = int(G.current_player(s))
  legal = _LEGAL(s)
  return int(_RULE(_INFO(s, p), legal, jax.random.key(key))), np.asarray(legal)


def _caller_state(first):
  """The Caller choosing trump after `first` (three cards).

  Args:
    first: three cards.

  Returns:
    The state.
  """
  s = G.new_initial_state()
  for a in first:
    s = _APPLY(s, jnp.int32(a))
  assert int(s.phase) == game.PHASE_CHOOSE_TRUMP
  return s


def test_registered_by_name():
  assert isinstance(registry.build(elena.ARCH, {}), elena.ElenaPolicy)
  ev = eval_registry.build(elena.EVAL_NAME, ENV)
  assert isinstance(ev, match.Evaluator)


def test_trump_is_the_suit_of_most_points():
  s = _caller_state([c(HERZ, A), c(CARO, Z), c(CARO, U)])  # 11 vs 12
  assert _elena(s)[0] == game.FIRST_TRUMP_ACTION + CARO


def test_trump_ties_are_random():
  s = _caller_state([c(HERZ, A), c(CARO, A), c(PIK, U)])
  picks = {_elena(s, k)[0] for k in range(32)}
  assert picks == {
      game.FIRST_TRUMP_ACTION + HERZ,
      game.FIRST_TRUMP_ACTION + CARO,
  }


CALL = {
    n: game.FIRST_CALL_ACTION + i - 1
    for i, n in enumerate(game.CONTRACT_NAMES)
    if i
}
PASS, FLECK = game.PASS_ACTION, game.FLECK_ACTION
_WEAK0 = [c(PIK, U), c(PIK, O), c(KREUZ, U), c(KREUZ, O), c(CARO, U)]


@pytest.mark.parametrize(
    "hand0,trump,want",
    [
        # All five trumps: Hermort; a whole other suit: Jodler.
        ([c(HERZ, r) for r in range(5)], HERZ, CALL["Hermort"]),
        ([c(CARO, r) for r in range(5)], HERZ, CALL["Jodler"]),
        # As, Zehn, Koenig of trump and another As: Bauern.
        ([c(HERZ, A), c(HERZ, Z), c(HERZ, K), c(CARO, A), c(PIK, U)], HERZ,
         CALL["Bauern"]),
        # Four Asse, one safe trump take: Lomt.
        ([c(HERZ, A), c(CARO, A), c(PIK, A), c(KREUZ, A), c(HERZ, U)], CARO,
         CALL["Lomt"]),
        # Three safe trump takes alone, the Caller's side: Schnapser.
        ([c(HERZ, A), c(HERZ, Z), c(HERZ, K), c(CARO, U), c(PIK, U)], HERZ,
         CALL["Schnapser"]),
        # As and Koenig (no Zehn): one safe take; pass.
        ([c(HERZ, A), c(HERZ, K), c(HERZ, O), c(CARO, Z), c(PIK, U)], HERZ,
         PASS),
    ],
)
def test_callers_bid(hand0, trump, want):
  a, legal = _elena(_deal([hand0], trump))
  assert legal[a] and a == want


def test_other_side_flecks_normal_and_callers_side_calls_schnapser():
  strong = [c(HERZ, A), c(HERZ, Z), c(HERZ, K), c(CARO, Z), c(PIK, Z)]
  rest = [c(CARO, O), c(CARO, K), c(PIK, K), c(KREUZ, K), c(HERZ, U)]
  # Seat 1 after seat 0's pass: fleck the Normal game.
  s = _deal([_WEAK0, strong], HERZ, bids=[PASS])
  assert _elena(s)[0] == FLECK
  # Seat 2 (the Caller's partner) after two passes: Schnapser.
  s = _deal([_WEAK0, rest, strong], HERZ, bids=[PASS, PASS])
  assert _elena(s)[0] == CALL["Schnapser"]


@pytest.mark.parametrize(
    "hand1,want",
    [
        # Three trumps: fleck.
        ([c(HERZ, Z), c(HERZ, O), c(HERZ, U), c(CARO, O), c(CARO, K)], FLECK),
        # Two trumps with the Zehn: fleck.
        ([c(HERZ, Z), c(HERZ, U), c(CARO, O), c(CARO, K), c(PIK, Z)], FLECK),
        # Two trumps without the Zehn: pass.
        ([c(HERZ, O), c(HERZ, U), c(CARO, O), c(CARO, K), c(PIK, Z)], PASS),
    ],
)
def test_fleck_an_opposing_bauern(hand1, want):
  s = _deal([_WEAK0, hand1], HERZ, bids=[CALL["Bauern"]])
  assert _elena(s)[0] == want


def _to_play(hands, trump):
  """Dealt, `trump` chosen, four passes: seat 0 leads the Normal game.

  Args:
    hands: the hands.
    trump: the trump suit.

  Returns:
    The state.
  """
  s = _deal(hands, trump, bids=[PASS] * 4)
  assert int(s.phase) == game.PHASE_PLAY and int(G.current_player(s)) == 0
  return s


def test_lead_prefers_a_boss_trump():
  s = _to_play(
      [[c(CARO, A), c(HERZ, A), c(PIK, U), c(KREUZ, U), c(CARO, U)]], HERZ
  )
  assert _elena(s)[0] == c(HERZ, A)
  # No boss trump: the boss of another suit.
  s = _to_play(
      [[c(CARO, A), c(HERZ, Z), c(PIK, U), c(KREUZ, U), c(CARO, U)]], HERZ
  )
  assert _elena(s)[0] == c(CARO, A)


def test_lead_boss_after_the_higher_cards_are_played():
  # Seat 0 leads its lowest card, the Caro Unter (it holds no boss card).
  hands = [
      [c(CARO, U), c(PIK, O), c(PIK, K), c(HERZ, O), c(HERZ, K)],
      [c(CARO, Z), c(CARO, A), c(KREUZ, A), c(PIK, U), c(HERZ, U)],
      [c(CARO, O), c(CARO, K), c(KREUZ, U), c(KREUZ, O), c(PIK, Z)],
      [c(KREUZ, Z), c(KREUZ, K), c(PIK, A), c(HERZ, Z), c(HERZ, A)],
  ]
  s = _to_play(hands, PIK)
  a, _ = _elena(s)
  assert a == c(CARO, U)
  # Seat 1 beats it with the Caro As, seat 2 follows, seat 3 trumps.
  for card in (c(CARO, U), c(CARO, A), c(CARO, K), c(PIK, A)):
    s = _APPLY(s, jnp.int32(card))
  # Seat 3 leads: its Kreuz Zehn is no boss (the Kreuz As is still out),
  # its Herz As is.
  assert int(G.current_player(s)) == 3
  assert _elena(s)[0] == c(HERZ, A)


def test_lead_lowest_non_trump_first():
  s = _to_play(
      [[c(HERZ, U), c(CARO, U), c(PIK, Z), c(KREUZ, Z), c(CARO, Z)]], HERZ
  )
  assert _elena(s)[0] == c(CARO, U)


def test_lead_declares_a_pair():
  # Lowest card: the Caro Ober, with its Koenig: declare 20 leading it.
  s = _to_play(
      [[c(CARO, O), c(CARO, K), c(PIK, Z), c(KREUZ, Z), c(HERZ, Z)]], HERZ
  )
  a, legal = _elena(s)
  assert legal[a] and a == game.FIRST_DECLARE_ACTION + 2 * CARO + 1


def test_follow_plays_the_lowest_legal_card():
  s = _to_play(
      [[c(CARO, U), c(PIK, O), c(PIK, K), c(HERZ, O), c(HERZ, K)],
       [c(CARO, Z), c(CARO, A), c(KREUZ, A), c(PIK, U), c(HERZ, U)]],
      PIK,
  )
  s = _APPLY(s, jnp.int32(c(CARO, U)))
  a, legal = _elena(s)
  # Seat 1 must follow Caro higher: Zehn and As are legal; the Zehn.
  assert legal[c(CARO, Z)] and legal[c(CARO, A)]
  assert a == c(CARO, Z)


@pytest.mark.parametrize("seed", [0, 1])
def test_rules_are_always_legal_in_random_games(seed):
  """Elena on seats chosen per deal, uniform play elsewhere, many deals."""
  rng = np.random.default_rng(seed)
  chance = jax.jit(G.chance_outcomes)
  n_decisions = 0
  kinds = set()
  for g in range(40):
    elena_seats = set(rng.choice(4, size=rng.integers(1, 5), replace=False))
    s = G.new_initial_state()
    while not bool(G.is_terminal(s)):
      if bool(G.is_chance_node(s)):
        p = np.asarray(chance(s), np.float64)
        a = int(rng.choice(len(p), p=p / p.sum()))
      else:
        p = int(G.current_player(s))
        legal = np.asarray(_LEGAL(s))
        if p in elena_seats:
          a, _ = _elena(s, key=1000 * g + n_decisions)
          assert legal[a], (g, int(s.phase), a, np.flatnonzero(legal))
          kinds.add(int(s.phase))
          n_decisions += 1
        else:
          a = int(rng.choice(np.flatnonzero(legal)))
      s = _APPLY(s, jnp.int32(a))
  assert n_decisions > 200
  assert {game.PHASE_BIDDING, game.PHASE_PLAY} <= kinds


def test_policy_beats_uniform():
  x64.enable_x64()
  ev = eval_registry.build(elena.EVAL_NAME, ENV)
  uni = match.Entrant(
      "uniform",
      uniform.UniformPolicy("schnaps", schnaps.NUM_ACTIONS, 4),
      {},
      types.SAMPLE,
  )
  res = ev.head_to_head(elena.entrant(), uni, 64, jax.random.key(0))
  print(f"elena vs uniform: {res.mean.mean:+.2f} +- {res.mean.se:.2f}")
  assert res.mean.mean > 0


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-s"]))
