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

"""Tests for the schnaps adapter."""

import json
import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui.adapters import schnaps_adapter

_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_SEATS = tuple(
    adapter.Viewpoint(seat=seat, omniscient=False) for seat in range(4)
)
_ADAPTER = schnaps_adapter.ADAPTER
_GAMES = 60


def _states(seed: int):
  """Every state of one seeded random deal, the first included."""
  game = pyspiel.load_game("schnaps")
  state = game.new_initial_state()
  rng = random.Random(seed)
  yield state
  while not state.is_terminal():
    state = state.child(rng.choice(state.legal_actions()))
    yield state


class SchnapsAdapterTest(absltest.TestCase):

  def test_card_codes(self):
    self.assertEqual(schnaps_adapter.card_code(0), "Jh")  # Herz Unter
    self.assertEqual(schnaps_adapter.card_code(2), "Kh")  # Herz Koenig
    self.assertEqual(schnaps_adapter.card_code(9), "Ad")  # Caro As
    self.assertEqual(schnaps_adapter.card_code(13), "Ts")  # Pik Zehn
    self.assertEqual(schnaps_adapter.card_code(16), "Qc")  # Kreuz Ober
    self.assertEqual(schnaps_adapter.card_name(2), "Herz König")

  def test_views_hide_other_hands(self):
    phases = set()
    for seed in range(_GAMES):
      for state in _states(seed):
        oracle = _ADAPTER.present(state, _ORACLE)
        phases.add(oracle["phase"])
        json.dumps(oracle)
        for seat, viewpoint in enumerate(_SEATS):
          view = _ADAPTER.present(state, viewpoint)
          for entry in view["seats"]:
            full = oracle["seats"][entry["seat"]]["cards"]
            if entry["seat"] == seat:
              self.assertEqual(entry["cards"], full)
            else:
              self.assertEqual(entry["cards"], [None] * len(full))
        spectator = _ADAPTER.present(state, _SPECTATOR)
        for entry in spectator["seats"]:
          self.assertTrue(all(card is None for card in entry["cards"]))
    self.assertEqual(
        phases, {"deal", "trump", "bidding", "play", "terminal"}
    )

  def test_last_trick_winner_leads_next(self):
    checked = 0
    for seed in range(_GAMES):
      for state in _states(seed):
        view = _ADAPTER.present(state, _ORACLE)
        last = view["lastTrick"]
        if last is None or view["phase"] != "play" or view["trick"]:
          continue
        self.assertLen(last["cards"], 4)
        self.assertEqual(last["winner"], view["toMove"])
        checked += 1
    self.assertGreater(checked, 0)

  def test_hints_cover_every_legal_action(self):
    for seed in range(_GAMES):
      for state in _states(seed):
        if state.is_terminal() or state.is_chance_node():
          continue
        actions = state.legal_actions()
        hints = _ADAPTER.describe_actions(state, actions, _ORACLE)
        self.assertCountEqual(hints.keys(), actions)
        cards = [h["card"] for h in hints.values() if "card" in h]
        self.assertLen(set(cards), len(cards))
        for action in actions:
          self.assertIsNotNone(_ADAPTER.public_action_label(state, action))

  def test_deal_is_hidden(self):
    state = pyspiel.load_game("schnaps").new_initial_state()
    self.assertIsNone(_ADAPTER.public_action_label(state, 0))


if __name__ == "__main__":
  absltest.main()
