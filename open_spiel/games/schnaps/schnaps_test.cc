// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/schnaps/schnaps.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace schnaps {
namespace {

void BasicSchnapsTests() {
  testing::LoadGameTest("schnaps");
  testing::ChanceOutcomesTest(*LoadGame("schnaps"));
  testing::RandomSimTest(*LoadGame("schnaps"), 100);
  testing::RandomSimTest(*LoadGame("schnaps(max_flecks=0)"), 20);
  testing::RandomSimTest(*LoadGame("schnaps(max_flecks=3)"), 20);
}

// Deal: P0 = all hearts (cards 0..4), P1 = caro, P2 = pik, P3 = kreuz.
// Trump = hearts. Cards are dealt in id order, which matches the dealing
// order (3 to P0, trump, then 2 to P0, 5 to P1, 5 to P2, 5 to P3).
std::unique_ptr<State> DealSortedHands() {
  std::shared_ptr<const Game> game = LoadGame("schnaps");
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action a = 0; a < 3; ++a) state->ApplyAction(a);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(kFirstTrumpAction + 0);  // hearts
  for (Action a = 3; a < kNumCards; ++a) state->ApplyAction(a);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);  // bidding starts with the Caller
  return state;
}

void HermortTest() {
  std::unique_ptr<State> state = DealSortedHands();
  // P0 holds the 5 trumps, so Hermort is available to him.
  const std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(),
                             kFirstCallAction + kHermort - 1) != legal.end());
  state->ApplyAction(kFirstCallAction + kHermort - 1);
  for (int i = 0; i < 3; ++i) state->ApplyAction(kPassAction);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  const std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 24.0);
  SPIEL_CHECK_EQ(returns[1], -24.0);
  SPIEL_CHECK_EQ(returns[2], 24.0);
  SPIEL_CHECK_EQ(returns[3], -24.0);
}

void NormalGameAndFollowTest() {
  std::unique_ptr<State> state = DealSortedHands();
  for (int i = 0; i < 4; ++i) state->ApplyAction(kPassAction);  // Normal
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // 5 hearts + declare 40 leading the King or the Ober of hearts.
  SPIEL_CHECK_EQ(state->LegalActions().size(), 7);
  state->ApplyAction(0);  // P0 leads Unter of hearts (trump)
  // P1 holds no hearts at all: free to play any of his 5 cards.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->LegalActions().size(), 5);
}

}  // namespace
}  // namespace schnaps
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::schnaps::BasicSchnapsTests();
  open_spiel::schnaps::HermortTest();
  open_spiel::schnaps::NormalGameAndFollowTest();
}
