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

"""Schnaps card planes over the envs.schnaps InfoState: f32[70, 20].

Importing this module registers the encoder; encoders.registry.LAZY names
this module, so a lookup of the name imports it (no import by the caller).

One column per card (card = suit * 5 + rank, suits Herz Caro Pik Kreuz,
ranks Unter Ober Koenig Zehn As), one row per channel (CHANNELS), from one
seat's view; seats are relative to the viewer (rel 0 the viewer, 1 the next
seat to play after it, 2 its partner, 3 the seat before it). Only what the
seat may know is encoded: its hand (and, for the Caller, which three cards
came first), trump, the public bidding, every played card (who played it,
in which trick, whether it was led or declared) and the voids those plays
reveal. Partners share no hidden information.

  24 card planes: my_hand, my_first_cards, played_by_rel0..3,
    played_in_trick0..4, led_card, declared_card, in_current_trick,
    winning_current_trick, is_trump, points_over11, rank_over4,
    follows_led_suit, void_rel1..3, my_pairs, unseen.
  46 broadcast planes (one value over every card): phase_trump,
    phase_bidding, phase_play, terminal, actor_rel0..3, caller_rel0..3,
    trump0..3, trump_active, contract0..5, declarer_rel0..3 (a called
    contract's declarer), last_actor_rel0..3, flecks_over6, passes_over3,
    any_call, score_us_over66, score_them_over66, pending_us_over66,
    pending_them_over66, tricks_us_over5, tricks_them_over5, leader_rel0..3,
    n_in_trick_over4, trick_num_over5.

Voids: a follower that did not follow the led suit is void in it; one that
also did not play trump (in a contract with trump) is void in trump too,
since a player void in the led suit must trump.

``waiting_row`` is a value-row predicate (learners.pg PGConfig.value_rows)
for self-play over all four seats with a round-robin row clock (row r
belongs to seat r % 4; a seat not on move waits on its row): at the ply
where seat Y is about to move after seat X moved, the seats strictly
between X and Y in seat order (all three others when X is Y: the Caller's
first bid right after choosing trump) are the seats such a clock reaches
while Y has yet to move. Left out: the trump choice, the first decision of
a deal (where the clock stands then depends on the previous deal, not on
this InfoState).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import schnaps

F32 = jnp.float32
I32 = jnp.int32
NAME = "schnaps_card_planes"
N_CARD = game.NUM_CARDS
N_PLAYER = game.NUM_PLAYERS
N_SUIT = game.NUM_SUITS
N_RANK = game.NUM_RANKS
N_TRICK = game.NUM_TRICKS
N_CONTRACT = len(game.CONTRACT_POINTS)

CARD_PLANES = (
    ("my_hand", "my_first_cards")
    + tuple(f"played_by_rel{k}" for k in range(N_PLAYER))
    + tuple(f"played_in_trick{t}" for t in range(N_TRICK))
    + ("led_card", "declared_card", "in_current_trick", "winning_current_trick")
    + ("is_trump", "points_over11", "rank_over4", "follows_led_suit")
    + tuple(f"void_rel{k}" for k in (1, 2, 3))
    + ("my_pairs", "unseen")
)
BROADCAST_PLANES = (
    ("b_phase_trump", "b_phase_bidding", "b_phase_play", "b_terminal")
    + tuple(f"b_actor_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_caller_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_trump{s}" for s in range(N_SUIT))
    + ("b_trump_active",)
    + tuple(f"b_contract{c}" for c in range(N_CONTRACT))
    + tuple(f"b_declarer_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_last_actor_rel{k}" for k in range(N_PLAYER))
    + ("b_flecks_over6", "b_passes_over3", "b_any_call")
    + ("b_score_us_over66", "b_score_them_over66")
    + ("b_pending_us_over66", "b_pending_them_over66")
    + ("b_tricks_us_over5", "b_tricks_them_over5")
    + tuple(f"b_leader_rel{k}" for k in range(N_PLAYER))
    + ("b_n_in_trick_over4", "b_trick_num_over5")
)
CHANNELS = CARD_PLANES + BROADCAST_PLANES
N_CHANNEL = len(CHANNELS)
assert N_CHANNEL == 70 and len(set(CHANNELS)) == N_CHANNEL


def _rel_onehot(seat, viewer):
  """bool[4]: `seat` relative to `viewer`; all False when seat < 0.

  Args:
    seat: i32 seat or -1.
    viewer: i32 seat.

  Returns:
    bool[4].
  """
  rel = (seat - viewer) % N_PLAYER
  return (jnp.arange(N_PLAYER) == rel) & (seat >= 0)


def _plays(info, trump_on, trump):
  """The facts of the cards played so far.

  Args:
    info: one seat's InfoState (envs.schnaps).
    trump_on: whether trump counts in the contract.
    trump: the trump suit.

  Returns:
    A dict of arrays (see the keys).
  """
  n_played = info["n_played"].astype(I32)
  plays = info["plays"].astype(I32)
  play_seat = info["play_seat"].astype(I32)
  pos = jnp.arange(N_CARD, dtype=I32)
  live = pos < n_played
  card_at = jnp.where(live, plays, 0)
  trick_at = pos // N_PLAYER
  place_at = pos % N_PLAYER
  suit_at = game.card_suit(card_at)
  rank_at = game.card_rank(card_at)
  led_at = suit_at[trick_at * N_PLAYER]
  is_trump_at = trump_on & (suit_at == trump)
  key = jnp.where(
      is_trump_at, 100 + rank_at, jnp.where(suit_at == led_at, rank_at, -1)
  )
  key = jnp.where(live, key, -2)
  idx = (
      jnp.arange(N_TRICK, dtype=I32)[:, None] * N_PLAYER
      + jnp.arange(N_PLAYER, dtype=I32)[None, :]
  )
  widx = idx[jnp.arange(N_TRICK), jnp.argmax(key[idx], axis=1)]
  winner = play_seat[widx]
  complete = (jnp.arange(N_TRICK) + 1) * N_PLAYER <= n_played
  target = jnp.where(live, plays, N_CARD)
  seat_of = jnp.full((N_CARD,), -1, I32).at[target].set(play_seat, mode="drop")
  trick_of = jnp.full((N_CARD,), -1, I32).at[target].set(trick_at, mode="drop")
  led_of = (
      jnp.zeros((N_CARD,), jnp.bool_).at[target].set(place_at == 0, mode="drop")
  )
  declared_of = (
      jnp.zeros((N_CARD,), jnp.bool_)
      .at[target]
      .set(info["declared"], mode="drop")
  )
  # Voids: not following the led suit; then not trumping, void in trump too.
  off = live & (place_at != 0) & (suit_at != led_at)
  no_trump = off & trump_on & ~is_trump_at
  void = jnp.zeros((N_PLAYER * N_SUIT,), jnp.bool_)
  void = void.at[jnp.where(off, play_seat * N_SUIT + led_at, N_PLAYER * N_SUIT)].set(
      True, mode="drop"
  )
  void = void.at[
      jnp.where(no_trump, play_seat * N_SUIT + jnp.maximum(trump, 0), N_PLAYER * N_SUIT)
  ].set(True, mode="drop")
  trick_num = n_played // N_PLAYER
  cur = jnp.minimum(trick_num, N_TRICK - 1)
  return dict(
      n_played=n_played,
      seat_of=seat_of,
      trick_of=trick_of,
      led_of=led_of,
      declared_of=declared_of,
      played=seat_of >= 0,
      void=void.reshape(N_PLAYER, N_SUIT),
      winner=winner,
      complete=complete,
      trick_num=trick_num,
      n_in_trick=n_played % N_PLAYER,
      led_suit=led_at[jnp.minimum(trick_num * N_PLAYER, N_CARD - 1)],
      win_card=card_at[widx[cur]],
      last_seat=play_seat[jnp.maximum(n_played - 1, 0)],
      first_seat=play_seat[jnp.minimum(trick_num * N_PLAYER, N_CARD - 1)],
  )


def _facts(info):
  """The phase, the seats on move and the play facts of one InfoState.

  Args:
    info: one seat's InfoState (envs.schnaps).

  Returns:
    A dict of arrays (see the keys).
  """
  viewer = info["seat"].astype(I32)
  phase = info["phase"].astype(I32)
  trump = info["trump"].astype(I32)
  contract = info["contract"].astype(I32)
  declarer = info["declarer"].astype(I32)
  n_bids = info["n_bids"].astype(I32)
  trump_on = (trump >= 0) & (contract != game.LOMT)
  p = _plays(info, trump_on, trump)
  choosing = phase == game.PHASE_CHOOSE_TRUMP
  bidding = phase == game.PHASE_BIDDING
  playing = phase == game.PHASE_PLAY
  first_leader = jnp.where(contract == game.LOMT, declarer, 0)
  play_actor = jnp.where(
      p["n_played"] == 0,
      first_leader,
      jnp.where(
          p["n_in_trick"] > 0,
          (p["last_seat"] + 1) % N_PLAYER,
          p["winner"][jnp.clip(p["trick_num"] - 1, 0, N_TRICK - 1)],
      ),
  )
  actor = jnp.select(
      [choosing, bidding, playing], [0, n_bids % N_PLAYER, play_actor], -1
  )
  last_bidder = jnp.where(n_bids > 0, (n_bids - 1) % N_PLAYER, 0)
  prev = jnp.select(
      [
          bidding,
          playing & (p["n_played"] > 0),
          playing,
      ],
      [last_bidder, p["last_seat"], last_bidder],
      -1,
  )
  leader = jnp.where(playing & (p["n_in_trick"] > 0), p["first_seat"], -1)
  return dict(
      p,
      viewer=viewer,
      phase=phase,
      trump=trump,
      contract=contract,
      declarer=declarer,
      trump_on=trump_on,
      playing=playing,
      actor=actor,
      prev=prev,
      leader=leader,
  )


def card_planes(info) -> jax.Array:
  """The card planes of one seat's InfoState (module doc).

  Args:
    info: one seat's InfoState (envs.schnaps).

  Returns:
    f32[70, 20].
  """
  f = _facts(info)
  viewer, trump = f["viewer"], f["trump"]
  cards = jnp.arange(N_CARD, dtype=I32)
  suits = game.card_suit(cards)
  ranks = game.card_rank(cards)
  hand = info["hand"]
  played = f["played"]
  in_trick = (
      played
      & (f["trick_of"] == f["trick_num"])
      & f["playing"]
      & (f["n_in_trick"] > 0)
  )
  played_rel = (f["seat_of"] - viewer) % N_PLAYER
  by_suit = hand.reshape(N_SUIT, N_RANK)
  pair = by_suit[:, game.KOENIG] & by_suit[:, game.OBER]
  my_pairs = pair[suits] & ((ranks == game.KOENIG) | (ranks == game.OBER))
  void = f["void"]
  card = [
      hand,
      info["first_cards"],
      *[played & (played_rel == k) for k in range(N_PLAYER)],
      *[played & (f["trick_of"] == t) for t in range(N_TRICK)],
      f["led_of"] & played,
      f["declared_of"] & played,
      in_trick,
      (cards == f["win_card"]) & in_trick,
      f["trump_on"] & (suits == trump),
      jnp.asarray(game.RANK_POINTS, F32)[ranks] / 11.0,
      ranks.astype(F32) / (N_RANK - 1),
      in_trick.any() & (suits == f["led_suit"]),
      *[void[(viewer + k) % N_PLAYER][suits] for k in (1, 2, 3)],
      my_pairs,
      ~hand & ~played,
  ]
  us = viewer % 2
  score = info["team_score"].astype(F32) / game.WINNING_POINTS
  pending = info["pending"].astype(F32) / game.WINNING_POINTS
  tricks = info["team_tricks"].astype(F32) / N_TRICK
  called = f["contract"] != game.NORMAL
  broadcast = jnp.concatenate([
      jnp.stack([
          f["phase"] == game.PHASE_CHOOSE_TRUMP,
          f["phase"] == game.PHASE_BIDDING,
          f["playing"],
          f["phase"] == game.PHASE_OVER,
      ]).astype(F32),
      _rel_onehot(f["actor"], viewer).astype(F32),
      _rel_onehot(jnp.int32(0), viewer).astype(F32),
      ((jnp.arange(N_SUIT) == trump) & (trump >= 0)).astype(F32),
      f["trump_on"].astype(F32)[None],
      (jnp.arange(N_CONTRACT) == f["contract"]).astype(F32),
      (_rel_onehot(f["declarer"], viewer) & called).astype(F32),
      _rel_onehot(info["last_actor"].astype(I32), viewer).astype(F32),
      (info["flecks"].astype(F32) / game.MAX_MAX_FLECKS)[None],
      (info["passes"].astype(F32) / 3.0)[None],
      info["any_call"].astype(F32)[None],
      jnp.stack([score[us], score[1 - us], pending[us], pending[1 - us]]),
      jnp.stack([tricks[us], tricks[1 - us]]),
      _rel_onehot(f["leader"], viewer).astype(F32),
      (
          jnp.where(f["playing"], f["n_in_trick"], 0).astype(F32)
          / (N_PLAYER - 1)
      )[None],
      (jnp.minimum(f["trick_num"], N_TRICK).astype(F32) / N_TRICK)[None],
  ])
  planes = jnp.stack([jnp.asarray(c).astype(F32) for c in card])
  out = jnp.concatenate([
      planes,
      jnp.broadcast_to(broadcast[:, None], (broadcast.shape[0], N_CARD)),
  ])
  assert out.shape == (N_CHANNEL, N_CARD), out.shape
  return out


def waiting_row(info) -> jax.Array:
  """bool: a value row of `info`'s seat (module doc, round-robin clock).

  Args:
    info: one seat's InfoState (envs.schnaps).

  Returns:
    bool[].
  """
  f = _facts(info)
  x, y, seat = f["prev"], f["actor"], f["viewer"]
  gap = (y - x - 1) % N_PLAYER
  k = (seat - x - 1) % N_PLAYER
  return (x >= 0) & (y >= 0) & (seat != y) & (k < gap)


CARD_PLANES_ENCODER = registry.register(
    registry.Encoder(
        name=NAME,
        game=schnaps.SchnapsEnv.name,
        shape=(N_CHANNEL, N_CARD),
        dtype=F32,
        fn=card_planes,
        reference="per-card planes of one seat's view (module doc)",
    )
)
