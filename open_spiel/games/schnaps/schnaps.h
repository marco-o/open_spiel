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

#ifndef OPEN_SPIEL_GAMES_SCHNAPS_SCHNAPS_H_
#define OPEN_SPIEL_GAMES_SCHNAPS_SCHNAPS_H_

// Bauernschnapsen: 4 players in 2 partnerships, 20 cards.
//
// Seats: player 0 is the "Caller" (left of the dealer) and sits opposite
// player 2; players 1 and 3 are the other partnership. Play is clockwise
// (0 -> 1 -> 2 -> 3).
//
// Phases of a deal:
//   1. Chance deals 3 cards to the Caller (player 0).
//   2. The Caller chooses the trump suit (atout).
//   3. Chance deals the remaining 17 cards (2 to the Caller, 5 to the others).
//      (Cutting / knocking only changes the physical dealing order, not the
//      distribution, so it is not modelled.)
//   4. Bidding: pass / fleck / call a higher contract.
//   5. Play of up to 5 tricks.
//
// Parameters:
//   max_flecks (int, default 2): max number of flecks (doublings) that can be
//                                given to a single contract.
//   normal_only (bool, default false): the bidding allows only passes, so
//                                every deal is a Normal game.
//
// Returns are zero-sum: the winning partnership gets +value, the other -value,
// where value = contract_points * 2^flecks.
//
// This implements ONE deal. The set (31 points counted downwards) and the
// match (3 sets) layers are not modelled.

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace schnaps {

inline constexpr int kNumPlayers = 4;
inline constexpr int kNumSuits = 4;  // Herz, Caro, Pik, Kreuz
inline constexpr int kNumRanks = 5;
inline constexpr int kNumCards = kNumSuits * kNumRanks;  // 20
inline constexpr int kHandSize = 5;
inline constexpr int kDefaultMaxFlecks = 2;

// Ranks in ascending strength.
enum Rank { kUnter = 0, kOber = 1, kKoenig = 2, kZehn = 3, kAs = 4 };

// Card points by rank: Unter 2, Ober 3, Koenig 4, Zehn 10, As 11 (120 total).
inline constexpr std::array<int, kNumRanks> kRankPoints = {2, 3, 4, 10, 11};

enum Contract {
  kNormal = 0,
  kSchnapser,
  kLomt,
  kBauern,
  kJodler,
  kHermort,
  kNumContracts
};
inline constexpr std::array<int, kNumContracts> kContractPoints = {3,  6,  9,
                                                                   12, 18, 24};

// ---------------------------------------------------------------------------
// Action encoding
//   0..19   a card (suit * 5 + rank): chance deals it / a player plays it
//   20..27  "declare 20/40 and lead King (even) or Ober (odd) of suit":
//           20 + 2 * suit + (0 = King, 1 = Ober)
//   28..31  choose trump suit
//   32      pass
//   33      fleck
//   34..38  call contract: Schnapser, Lomt, Bauern, Jodler, Hermort
// ---------------------------------------------------------------------------
inline constexpr int kFirstDeclareAction = kNumCards;                 // 20
inline constexpr int kNumDeclareActions = 2 * kNumSuits;              // 8
inline constexpr int kFirstTrumpAction = kFirstDeclareAction + 8;     // 28
inline constexpr int kPassAction = kFirstTrumpAction + kNumSuits;     // 32
inline constexpr int kFleckAction = kPassAction + 1;                  // 33
inline constexpr int kFirstCallAction = kFleckAction + 1;             // 34
inline constexpr int kNumDistinctActions = kFirstCallAction + 5;      // 39

enum class Phase {
  kDealCaller,   // chance: first 3 cards to player 0
  kChooseTrump,  // player 0 picks the trump
  kDealRest,     // chance: remaining 17 cards
  kBidding,
  kPlay,
  kTerminal
};

struct PlayedCard {
  Player player;
  int card;
};

class SchnapsState : public State {
 public:
  SchnapsState(std::shared_ptr<const Game> game, int max_flecks,
               bool normal_only);
  SchnapsState(const SchnapsState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kTerminal; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Helpers.
  std::vector<Action> UndealtCards() const;
  std::vector<int> Hand(Player player) const;
  std::string HandString(Player player) const;
  std::string PhaseName() const;
  Player DealRecipient() const;
  bool TrumpActive() const { return contract_ != kLomt; }
  bool HoldsWholeSuit(Player player, int suit) const;
  bool HoldsJodler(Player player) const;
  bool HoldsHermort(Player player) const;

  std::vector<Action> BiddingActions() const;
  std::vector<Action> PlayActions() const;

  void ApplyDeal(Action card);
  void ApplyTrump(Action action);
  void ApplyBid(Action action);
  void ApplyPlay(Action action);
  void EndBidding();
  void PlayCard(Player player, int card);
  void ResolveTrick();
  // Returns true if the deal ended.
  bool CheckGameEnd(bool after_trick);
  void FinishNormal(int winning_team);
  void FinishGame(int winning_team, int base_points);

  int max_flecks_;
  bool normal_only_;
  Phase phase_ = Phase::kDealCaller;

  // owner_[card] = player holding it, or -1 if not dealt yet.
  std::array<Player, kNumCards> owner_;
  std::array<bool, kNumCards> played_;
  std::vector<int> caller_first_cards_;  // the 3 cards seen before trump call
  int num_dealt_caller_ = 0;
  int num_dealt_rest_ = 0;
  int trump_ = -1;

  // Bidding state.
  int contract_ = kNormal;
  Player declarer_ = 0;     // who called the current contract (0 for Normal)
  Player last_actor_ = 0;   // last player to call or fleck (0 initially)
  Player bidder_ = 0;
  int flecks_ = 0;
  int passes_in_row_ = 0;
  bool any_action_ = false;  // any call/fleck since start of bidding

  // Play state.
  Player leader_ = 0;
  Player current_player_ = 0;
  std::vector<PlayedCard> trick_;
  std::array<int, kNumPlayers> tricks_won_{};
  std::array<int, 2> team_tricks_{};
  std::array<int, 2> team_score_{};
  std::array<int, 2> pending_decl_{};  // declared but no trick won yet
  int tricks_played_ = 0;
  Player last_trick_winner_ = -1;

  std::vector<double> returns_;
};

class SchnapsGame : public Game {
 public:
  explicit SchnapsGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kNumCards; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override;
  double MaxUtility() const override;
  std::optional<double> UtilitySum() const override { return 0; }
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override { return kNumCards; }

 private:
  int max_flecks_;
  bool normal_only_;
};

}  // namespace schnaps
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SCHNAPS_SCHNAPS_H_
