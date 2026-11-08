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
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace schnaps {
namespace {

const GameType kGameType{
    /*short_name=*/"schnaps",
    /*long_name=*/"Bauernschnapsen",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"max_flecks", GameParameter(kDefaultMaxFlecks)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<SchnapsGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

const char* const kRankStr[kNumRanks] = {"U", "O", "K", "Z", "A"};
const char kSuitChar[] = "hcpk";  // Herz, Caro, Pik, Kreuz
const char* const kContractNames[kNumContracts] = {
    "Normal", "Schnapser", "Lomt", "Bauern", "Jodler", "Hermort"};

int CardSuit(int card) { return card / kNumRanks; }
int CardRank(int card) { return card % kNumRanks; }
int MakeCard(int suit, int rank) { return suit * kNumRanks + rank; }
int TeamOf(Player p) { return p % 2; }

std::string SuitString(int suit) { return std::string(1, kSuitChar[suit]); }

std::string CardString(int card) {
  return absl::StrCat(kRankStr[CardRank(card)], SuitString(CardSuit(card)));
}

}  // namespace

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

SchnapsState::SchnapsState(std::shared_ptr<const Game> game, int max_flecks)
    : State(game), max_flecks_(max_flecks) {
  owner_.fill(-1);
  played_.fill(false);
}

std::unique_ptr<State> SchnapsState::Clone() const {
  return std::unique_ptr<State>(new SchnapsState(*this));
}

Player SchnapsState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kDealCaller:
    case Phase::kDealRest:
      return kChancePlayerId;
    case Phase::kChooseTrump:
      return 0;
    case Phase::kBidding:
      return bidder_;
    case Phase::kPlay:
      return current_player_;
    case Phase::kTerminal:
      return kTerminalPlayerId;
  }
  return kTerminalPlayerId;
}

std::vector<Action> SchnapsState::UndealtCards() const {
  std::vector<Action> cards;
  for (int c = 0; c < kNumCards; ++c) {
    if (owner_[c] == -1) cards.push_back(c);
  }
  return cards;
}

std::vector<int> SchnapsState::Hand(Player player) const {
  std::vector<int> hand;
  for (int c = 0; c < kNumCards; ++c) {
    if (owner_[c] == player && !played_[c]) hand.push_back(c);
  }
  return hand;
}

std::string SchnapsState::HandString(Player player) const {
  std::string s;
  for (int c : Hand(player)) {
    if (!s.empty()) s += " ";
    s += CardString(c);
  }
  return s;
}

std::string SchnapsState::PhaseName() const {
  switch (phase_) {
    case Phase::kDealCaller: return "DealCaller";
    case Phase::kChooseTrump: return "ChooseTrump";
    case Phase::kDealRest: return "DealRest";
    case Phase::kBidding: return "Bidding";
    case Phase::kPlay: return "Play";
    case Phase::kTerminal: return "Terminal";
  }
  return "?";
}

// Who receives the next dealt card.
Player SchnapsState::DealRecipient() const {
  if (phase_ == Phase::kDealCaller) return 0;
  const int k = num_dealt_rest_;  // 0..16
  if (k < 2) return 0;            // Caller gets 2 more
  return 1 + (k - 2) / 5;         // then 5 each to players 1, 2, 3
}

bool SchnapsState::HoldsWholeSuit(Player player, int suit) const {
  for (int r = 0; r < kNumRanks; ++r) {
    if (owner_[MakeCard(suit, r)] != player) return false;
  }
  return true;
}

bool SchnapsState::HoldsJodler(Player player) const {
  for (int s = 0; s < kNumSuits; ++s) {
    if (s != trump_ && HoldsWholeSuit(player, s)) return true;
  }
  return false;
}

bool SchnapsState::HoldsHermort(Player player) const {
  return HoldsWholeSuit(player, trump_);
}

// ----------------------------- Legal actions -------------------------------

std::vector<Action> SchnapsState::LegalActions() const {
  switch (phase_) {
    case Phase::kDealCaller:
    case Phase::kDealRest:
      return UndealtCards();
    case Phase::kChooseTrump: {
      std::vector<Action> actions;
      for (int s = 0; s < kNumSuits; ++s) {
        actions.push_back(kFirstTrumpAction + s);
      }
      return actions;
    }
    case Phase::kBidding:
      return BiddingActions();
    case Phase::kPlay:
      return PlayActions();
    case Phase::kTerminal:
      return {};
  }
  return {};
}

std::vector<std::pair<Action, double>> SchnapsState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  const std::vector<Action> cards = UndealtCards();
  const double prob = 1.0 / cards.size();
  std::vector<std::pair<Action, double>> outcomes;
  for (Action c : cards) outcomes.push_back({c, prob});
  return outcomes;
}

std::vector<Action> SchnapsState::BiddingActions() const {
  const Player p = bidder_;
  std::vector<Action> actions = {kPassAction};
  // Fleck: only against a call/fleck made by the opposing partnership.
  if (flecks_ < max_flecks_ && TeamOf(last_actor_) != TeamOf(p)) {
    actions.push_back(kFleckAction);
  }
  // Higher contract. Jodler / Hermort need the matching hand.
  for (int c = contract_ + 1; c < kNumContracts; ++c) {
    if (c == kJodler && !HoldsJodler(p)) continue;
    if (c == kHermort && !HoldsHermort(p)) continue;
    actions.push_back(kFirstCallAction + (c - 1));
  }
  return actions;
}

std::vector<Action> SchnapsState::PlayActions() const {
  const Player p = current_player_;
  const std::vector<int> hand = Hand(p);
  std::vector<Action> actions;

  if (trick_.empty()) {
    for (int c : hand) actions.push_back(c);
    // Declarations (20 / 40) are only relevant for point-race contracts.
    if (contract_ == kNormal || contract_ == kSchnapser) {
      for (int s = 0; s < kNumSuits; ++s) {
        const int king = MakeCard(s, kKoenig);
        const int ober = MakeCard(s, kOber);
        if (owner_[king] == p && !played_[king] && owner_[ober] == p &&
            !played_[ober]) {
          actions.push_back(kFirstDeclareAction + 2 * s);      // lead King
          actions.push_back(kFirstDeclareAction + 2 * s + 1);  // lead Ober
        }
      }
    }
    return actions;
  }

  const int led = CardSuit(trick_[0].card);
  int best_led_rank = -1;
  int best_trump_rank = -1;
  bool trump_played = false;
  for (const PlayedCard& pc : trick_) {
    const int s = CardSuit(pc.card);
    const int r = CardRank(pc.card);
    if (TrumpActive() && s == trump_) {
      trump_played = true;
      best_trump_rank = std::max(best_trump_rank, r);
    }
    if (s == led) best_led_rank = std::max(best_led_rank, r);
  }

  std::vector<int> follow;
  std::vector<int> trumps;
  for (int c : hand) {
    if (CardSuit(c) == led) follow.push_back(c);
    if (TrumpActive() && CardSuit(c) == trump_) trumps.push_back(c);
  }

  if (!follow.empty()) {
    // Must follow suit. Must also beat the best card of the led suit, unless
    // a trump (different from the led suit) has already been played.
    const bool already_trumped = TrumpActive() && trump_played && led != trump_;
    if (!already_trumped) {
      std::vector<Action> higher;
      for (int c : follow) {
        if (CardRank(c) > best_led_rank) higher.push_back(c);
      }
      if (!higher.empty()) return higher;
    }
    return std::vector<Action>(follow.begin(), follow.end());
  }

  if (TrumpActive() && !trumps.empty()) {
    // Void in the led suit: must trump, overtrumping if possible.
    std::vector<Action> higher;
    for (int c : trumps) {
      if (CardRank(c) > best_trump_rank) higher.push_back(c);
    }
    if (!higher.empty()) return higher;
    return std::vector<Action>(trumps.begin(), trumps.end());
  }

  return std::vector<Action>(hand.begin(), hand.end());
}

// ---------------------------- Applying actions -----------------------------

void SchnapsState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDealCaller:
    case Phase::kDealRest:
      ApplyDeal(action);
      return;
    case Phase::kChooseTrump:
      ApplyTrump(action);
      return;
    case Phase::kBidding:
      ApplyBid(action);
      return;
    case Phase::kPlay:
      ApplyPlay(action);
      return;
    case Phase::kTerminal:
      SpielFatalError("Cannot act in a terminal state.");
  }
}

void SchnapsState::ApplyDeal(Action card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  SPIEL_CHECK_EQ(owner_[card], -1);
  owner_[card] = DealRecipient();
  if (phase_ == Phase::kDealCaller) {
    caller_first_cards_.push_back(static_cast<int>(card));
    if (++num_dealt_caller_ == 3) phase_ = Phase::kChooseTrump;
  } else {
    if (++num_dealt_rest_ == kNumCards - 3) {
      phase_ = Phase::kBidding;
      bidder_ = 0;
    }
  }
}

void SchnapsState::ApplyTrump(Action action) {
  SPIEL_CHECK_GE(action, kFirstTrumpAction);
  SPIEL_CHECK_LT(action, kFirstTrumpAction + kNumSuits);
  trump_ = action - kFirstTrumpAction;
  phase_ = Phase::kDealRest;
}

void SchnapsState::ApplyBid(Action action) {
  const Player p = bidder_;
  if (action == kPassAction) {
    ++passes_in_row_;
  } else if (action == kFleckAction) {
    ++flecks_;
    last_actor_ = p;
    passes_in_row_ = 0;
    any_action_ = true;
  } else {
    SPIEL_CHECK_GE(action, kFirstCallAction);
    SPIEL_CHECK_LT(action, kFirstCallAction + 5);
    contract_ = static_cast<int>(action) - kFirstCallAction + 1;
    declarer_ = p;
    last_actor_ = p;
    flecks_ = 0;  // flecks apply to the contract they were given to
    passes_in_row_ = 0;
    any_action_ = true;
  }
  bidder_ = (p + 1) % kNumPlayers;
  // Bidding ends when the 3 other players have all passed after the last
  // call/fleck, or when all 4 pass without anything having been called.
  if (passes_in_row_ >= (any_action_ ? 3 : 4)) EndBidding();
}

void SchnapsState::EndBidding() {
  if (contract_ == kJodler || contract_ == kHermort) {
    // The declarer holds a guaranteed winning hand: resolved without play.
    FinishGame(TeamOf(declarer_), kContractPoints[contract_]);
    return;
  }
  leader_ = (contract_ == kLomt) ? declarer_ : 0;
  current_player_ = leader_;
  phase_ = Phase::kPlay;
}

void SchnapsState::ApplyPlay(Action action) {
  const Player p = current_player_;
  int card;
  if (action >= kFirstDeclareAction &&
      action < kFirstDeclareAction + kNumDeclareActions) {
    const int idx = static_cast<int>(action) - kFirstDeclareAction;
    const int suit = idx / 2;
    card = MakeCard(suit, idx % 2 == 0 ? kKoenig : kOber);
    const int points = (suit == trump_) ? 40 : 20;
    const int team = TeamOf(p);
    // Declarations only count once the partnership has won a trick.
    if (team_tricks_[team] > 0) {
      team_score_[team] += points;
    } else {
      pending_decl_[team] += points;
    }
    if (CheckGameEnd(/*after_trick=*/false)) return;
  } else {
    card = static_cast<int>(action);
  }
  PlayCard(p, card);
}

void SchnapsState::PlayCard(Player player, int card) {
  SPIEL_CHECK_EQ(owner_[card], player);
  SPIEL_CHECK_FALSE(played_[card]);
  played_[card] = true;
  trick_.push_back({player, card});
  if (trick_.size() < kNumPlayers) {
    current_player_ = (player + 1) % kNumPlayers;
    return;
  }
  ResolveTrick();
}

void SchnapsState::ResolveTrick() {
  // Does card `a` beat card `b`, where `b` is the currently winning card?
  auto beats = [this](int a, int b) {
    const bool ta = TrumpActive() && CardSuit(a) == trump_;
    const bool tb = TrumpActive() && CardSuit(b) == trump_;
    if (ta != tb) return ta;
    if (CardSuit(a) == CardSuit(b)) return CardRank(a) > CardRank(b);
    return false;
  };
  int win_idx = 0;
  int points = 0;
  for (int i = 0; i < kNumPlayers; ++i) {
    points += kRankPoints[CardRank(trick_[i].card)];
    if (i > 0 && beats(trick_[i].card, trick_[win_idx].card)) win_idx = i;
  }
  const Player winner = trick_[win_idx].player;
  const int team = TeamOf(winner);

  team_score_[team] += points;
  ++tricks_won_[winner];
  ++team_tricks_[team];
  if (pending_decl_[team] > 0) {  // declarations become valid now
    team_score_[team] += pending_decl_[team];
    pending_decl_[team] = 0;
  }
  last_trick_winner_ = winner;
  ++tricks_played_;
  trick_.clear();
  leader_ = winner;
  current_player_ = winner;
  CheckGameEnd(/*after_trick=*/true);
}

bool SchnapsState::CheckGameEnd(bool after_trick) {
  switch (contract_) {
    case kNormal:
      for (int t = 0; t < 2; ++t) {
        if (team_score_[t] >= 66) {
          FinishNormal(t);
          return true;
        }
      }
      if (after_trick && tricks_played_ == kHandSize) {
        FinishNormal(TeamOf(last_trick_winner_));  // last trick wins
        return true;
      }
      return false;
    case kSchnapser: {
      const int t = TeamOf(declarer_);
      if (team_score_[t] >= 66) {
        FinishGame(t, kContractPoints[kSchnapser]);
        return true;
      }
      if (after_trick && tricks_played_ == 3) {
        FinishGame(1 - t, kContractPoints[kSchnapser]);
        return true;
      }
      return false;
    }
    case kLomt:
    case kBauern: {
      if (!after_trick) return false;
      const int t = TeamOf(declarer_);
      if (last_trick_winner_ != declarer_) {  // someone else took a trick
        FinishGame(1 - t, kContractPoints[contract_]);
        return true;
      }
      if (tricks_played_ == kHandSize) {
        FinishGame(t, kContractPoints[contract_]);
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

void SchnapsState::FinishNormal(int winning_team) {
  const int loser = 1 - winning_team;
  int base;
  if (team_tricks_[loser] == 0) {
    base = 3;
  } else if (team_score_[loser] < 33) {
    base = 2;
  } else {
    base = 1;
  }
  FinishGame(winning_team, base);
}

void SchnapsState::FinishGame(int winning_team, int base_points) {
  const double value = static_cast<double>(base_points) * (1 << flecks_);
  returns_.assign(kNumPlayers, 0.0);
  for (Player p = 0; p < kNumPlayers; ++p) {
    returns_[p] = (TeamOf(p) == winning_team) ? value : -value;
  }
  phase_ = Phase::kTerminal;
}

std::vector<double> SchnapsState::Returns() const {
  if (!IsTerminal()) return std::vector<double>(kNumPlayers, 0.0);
  return returns_;
}

// ------------------------------- Strings -----------------------------------

std::string SchnapsState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal ", CardString(static_cast<int>(action)));
  }
  if (action < kNumCards) {
    return absl::StrCat("Play ", CardString(static_cast<int>(action)));
  }
  if (action < kFirstTrumpAction) {
    const int idx = static_cast<int>(action) - kFirstDeclareAction;
    const int suit = idx / 2;
    const int card = MakeCard(suit, idx % 2 == 0 ? kKoenig : kOber);
    return absl::StrCat("Declare ", suit == trump_ ? "40" : "20", " & lead ",
                        CardString(card));
  }
  if (action < kPassAction) {
    return absl::StrCat("Trump ", SuitString(static_cast<int>(action) -
                                              kFirstTrumpAction));
  }
  if (action == kPassAction) return "Pass";
  if (action == kFleckAction) return "Fleck";
  return absl::StrCat(
      "Call ", kContractNames[static_cast<int>(action) - kFirstCallAction + 1]);
}

std::string SchnapsState::ToString() const {
  std::string s = absl::StrCat("Phase: ", PhaseName(), "\n");
  if (trump_ >= 0) absl::StrAppend(&s, "Trump: ", SuitString(trump_), "\n");
  for (Player p = 0; p < kNumPlayers; ++p) {
    absl::StrAppend(&s, "P", p, ": ", HandString(p), "\n");
  }
  if (phase_ == Phase::kBidding || phase_ == Phase::kPlay ||
      phase_ == Phase::kTerminal) {
    absl::StrAppend(&s, "Contract: ", kContractNames[contract_], " by P",
                    declarer_, ", flecks=", flecks_, "\n");
  }
  if (phase_ == Phase::kPlay || phase_ == Phase::kTerminal) {
    absl::StrAppend(&s, "Trick:");
    for (const PlayedCard& pc : trick_) {
      absl::StrAppend(&s, " P", pc.player, ":", CardString(pc.card));
    }
    absl::StrAppend(&s, "\nScore: ", team_score_[0], "-", team_score_[1],
                    "  Tricks: ", team_tricks_[0], "-", team_tricks_[1], "\n");
  }
  return s;
}

std::string SchnapsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::string s = absl::StrCat("P", player, " hand:", HandString(player));
  if (trump_ >= 0) absl::StrAppend(&s, " trump:", SuitString(trump_));
  if (phase_ == Phase::kBidding) {
    absl::StrAppend(&s, " bidder:P", bidder_);
  }
  if (phase_ == Phase::kBidding || phase_ == Phase::kPlay ||
      phase_ == Phase::kTerminal) {
    absl::StrAppend(&s, " contract:", kContractNames[contract_], "/P",
                    declarer_, "/flecks", flecks_);
  }
  if (phase_ == Phase::kPlay || phase_ == Phase::kTerminal) {
    absl::StrAppend(&s, " trick:");
    for (const PlayedCard& pc : trick_) {
      absl::StrAppend(&s, "P", pc.player, "=", CardString(pc.card), ",");
    }
    absl::StrAppend(&s, " score:", team_score_[0], "-", team_score_[1],
                    " decl:", pending_decl_[0], "-", pending_decl_[1],
                    " tricks:", tricks_won_[0], tricks_won_[1], tricks_won_[2],
                    tricks_won_[3]);
  }
  return s;
}

std::string SchnapsState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::string s = absl::StrCat("P", player);
  // The Caller chooses trump after seeing only the first 3 cards.
  if (player == 0 && !caller_first_cards_.empty()) {
    std::vector<int> first = caller_first_cards_;
    std::sort(first.begin(), first.end());
    absl::StrAppend(&s, " first:");
    for (int c : first) absl::StrAppend(&s, CardString(c), ",");
  }
  absl::StrAppend(&s, " hand:");
  for (int c = 0; c < kNumCards; ++c) {
    if (owner_[c] == player) absl::StrAppend(&s, CardString(c), ",");
  }
  absl::StrAppend(&s, " acts:");
  for (const PlayerAction& pa : history_) {
    if (pa.player != kChancePlayerId) absl::StrAppend(&s, pa.action, ",");
  }
  return s;
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

SchnapsGame::SchnapsGame(const GameParameters& params)
    : Game(kGameType, params),
      max_flecks_(ParameterValue<int>("max_flecks")) {
  SPIEL_CHECK_GE(max_flecks_, 0);
  SPIEL_CHECK_LE(max_flecks_, 6);
}

std::unique_ptr<State> SchnapsGame::NewInitialState() const {
  return std::unique_ptr<State>(new SchnapsState(shared_from_this(),
                                                 max_flecks_));
}

double SchnapsGame::MaxUtility() const {
  return static_cast<double>(kContractPoints[kHermort]) * (1 << max_flecks_);
}

double SchnapsGame::MinUtility() const { return -MaxUtility(); }

int SchnapsGame::MaxGameLength() const {
  // 20 deals + 1 trump call + bidding + at most 20 card plays.
  // Bidding: at most 5 raises and max_flecks flecks per contract level, each
  // non-pass followed by at most 2 passes before the next, + closing passes.
  const int non_pass = 5 + 6 * max_flecks_;
  const int bidding = 3 * non_pass + 4;
  return kNumCards + 1 + bidding + kNumCards;
}

}  // namespace schnaps
}  // namespace open_spiel
