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

"""Schnaps Env over the JAX engine `schnaps`: two partnerships of two.

Seats 0..3 (0 and 2 against 1 and 3; seat 0 is the Caller), 39 actions with
OpenSpiel's ids (0..19 cards; 20..27 declare 20/40 and lead the Koenig or
Ober of a suit; 28..31 choose trump; 32 pass; 33 fleck; 34..38 call
Schnapser, Lomt, Bauern, Jodler, Hermort), zero-sum returns: the winners'
score (the contract's base, doubled per fleck) for both winners, its negative
for the losers. One deal is one episode: chance deals three cards to the
Caller, who chooses trump; chance deals the other seventeen; then the
bidding and up to 20 plays (none after Jodler or Hermort). The rules are
OpenSpiel's schnaps (the games package's schnaps module doc); the one option
is its game parameter ``max_flecks`` (``OPTIONS``).

INFOSTATE. One seat's view, a dict with these keys and dtypes
(``info_template``):

  seat         int8          the observing seat
  phase        int8          the games package's schnaps phase
  trump        int8          the trump suit, -1 until chosen
  contract     int8          the contract (0 Normal .. 5 Hermort)
  declarer     int8          its declarer (0 for the Normal game)
  last_actor   int8          the last seat to call or fleck (0 at first)
  flecks       int8          the flecks on the contract
  passes       int8          passes since the last call or fleck
  any_call     bool          whether anything was called or flecked
  bids         int8[B]       every bid in order, as action - 32; -1 pad
                             (B = schnaps.max_bids(max_flecks))
  n_bids       int8          the bids made
  first_cards  bool[20]      the Caller's first three cards, in seat 0's
                             view only (all False for another seat)
  dealt        bool[20]      the cards dealt to the seat
  hand         bool[20]      the cards it holds now
  plays        int8[20]      the cards in order of play; -1 pad
  play_seat    int8[20]      who played each of them; -1 pad
  declared     bool[20]      whether each was led with a declaration
  n_played     int8          the cards played
  team_score   int32[2]      each team's points (cards and valid
                             declarations)
  pending      int32[2]      each team's declarations awaiting its first
                             trick
  team_tricks  int8[2]       each team's tricks

That is what OpenSpiel's information-state string holds (the trump choice,
every bid and every card played are public, the deal is not), and the
encoder ``schnaps_info_state`` (encoders.schnaps) rebuilds the games
package's tensor from it bit for bit. The other hands are never in it.
``labels`` carries the deal (``deal_holder``, the seat dealt each card).
Every player action is public. ``world_from_info`` is None;
``hide_hidden_cards`` is the privacy hook.

SIDES. The two partnerships (``Sides.partnerships(4, 2, constant_sum=True)``:
{0, 2} against {1, 3}); a side's return is its seats'.

RULES IDENTITY. Declared (``declared_rules_sha``): the sha256 over canonical
JSON of RULES with ``max_flecks``, so the hash names OpenSpiel's schnaps at
that parameter, not a source file. The games package's schnaps_test checks
the engine move for move against OpenSpiel's; a rules change there bumps
RULES ``schema`` (so the hash) in the same change.

SERVING. Served (envs.registry) at every ``max_flecks``; a pyspiel history
is the raw history (the same chance and action ids).
"""

import hashlib
import json
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import schnaps
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = schnaps.NUM_CARDS
NUM_ACTIONS = schnaps.NUM_DISTINCT_ACTIONS
NUM_PLAYERS = schnaps.NUM_PLAYERS
#: OpenSpiel schnaps's game parameter at its default.
OPTIONS = {"max_flecks": schnaps.DEFAULT_MAX_FLECKS}
assert tuple(OPTIONS) == schnaps.PARAMS
RULES = {
    "schema": "schnaps-rules/2",
    "engine": "open_spiel schnaps (games/schnaps/schnaps.cc)",
    "game": "schnaps",
    "scoring": (
        "Normal 3/2/1 by the losers' tricks and points (66 to win, else the "
        "last trick), Schnapser 6, Lomt 9, Bauern 12, Jodler 18, Hermort 24, "
        "doubled per fleck; Unter 2, Ober 3, Koenig 4, Zehn 10, As 11"
    ),
}
#: Seats 0 and 2 against seats 1 and 3, zero-sum.
SIDES = sides_lib.Sides.partnerships(NUM_PLAYERS, 2, constant_sum=True)


def decisions_bound(max_flecks: int) -> int:
  """The most player decisions in a deal: trump, the bids, 20 plays.

  Args:
    max_flecks: the game parameter.

  Returns:
    The bound.
  """
  return 1 + schnaps.max_bids(max_flecks) + NUM_CARDS


def rules_sha(options: Mapping[str, int]) -> str:
  """The declared rules identity of schnaps at `options` (module doc).

  Args:
    options: the game parameter.

  Returns:
    The sha256 hex digest of RULES with the parameter, as canonical JSON.
  """
  params = {k: int(options[k]) for k in OPTIONS}
  if options.get("normal_only"):  # only when set: schnaps's hash is unchanged
    params["normal_only"] = 1
  rules = {**RULES, "params": params}
  return hashlib.sha256(
      json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def info_template(
    max_flecks: int = schnaps.DEFAULT_MAX_FLECKS,
) -> dict[str, np.ndarray]:
  """One seat's InfoState (unbatched) as zero arrays.

  Of its shapes and dtypes.

  Args:
    max_flecks: the game parameter (it sets the bid slots).

  Returns:
    {key: zero array}.
  """
  scalars = (
      "seat",
      "phase",
      "trump",
      "contract",
      "declarer",
      "last_actor",
      "flecks",
      "passes",
      "n_bids",
      "n_played",
  )
  return {
      **{k: np.zeros((), np.int8) for k in scalars},
      "any_call": np.zeros((), np.bool_),
      "bids": np.full((schnaps.max_bids(max_flecks),), -1, np.int8),
      "first_cards": np.zeros((NUM_CARDS,), np.bool_),
      "dealt": np.zeros((NUM_CARDS,), np.bool_),
      "hand": np.zeros((NUM_CARDS,), np.bool_),
      "plays": np.full((NUM_CARDS,), -1, np.int8),
      "play_seat": np.full((NUM_CARDS,), -1, np.int8),
      "declared": np.zeros((NUM_CARDS,), np.bool_),
      "team_score": np.zeros((2,), np.int32),
      "pending": np.zeros((2,), np.int32),
      "team_tricks": np.zeros((2,), np.int8),
  }


class SchnapsEnv(jax_game_env.JaxGameEnv):
  """Schnaps: 4 seats in 2 partnerships, 39 actions, zero-sum (module doc)."""

  name = "schnaps"
  max_len = decisions_bound(schnaps.DEFAULT_MAX_FLECKS)
  zero_sum = True
  sides = SIDES

  def __init__(self, **options):
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
      raise TypeError(
          f"schnaps: unknown options {unknown}; the options are "
          f"{sorted(OPTIONS)}"
      )
    self.options = options_check({**OPTIONS, **options})
    # registry.spec_of reads the options as attributes
    for k, v in self.options.items():
      setattr(self, k, v)
    self.max_len = decisions_bound(self.options["max_flecks"])
    super().__init__(schnaps.Schnaps(**self.options))
    assert self.num_players == NUM_PLAYERS and self.num_actions == NUM_ACTIONS
    assert self.max_chance_outcomes == NUM_CARDS

  @property
  def declared_rules_sha(self) -> str:
    """The rules identity: rules_sha of the options (module doc)."""
    return rules_sha(self.options)

  def info_state(self, s: schnaps.GameState, seat) -> dict[str, jax.Array]:
    """One seat's InfoState (module doc, INFOSTATE).

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      {key: array}.
    """
    seat = jnp.asarray(seat, jnp.int32)

    def i8(x):
      """`x` as int8.

      Args:
        x: An array.

      Returns:
        The int8 array.
      """
      return jnp.asarray(x, jnp.int8)

    dealt = s.holder == seat
    return {
        "seat": i8(seat),
        "phase": i8(s.phase),
        "trump": i8(s.trump),
        "contract": i8(s.contract),
        "declarer": i8(s.declarer),
        "last_actor": i8(s.last_actor),
        "flecks": i8(s.flecks),
        "passes": i8(s.passes),
        "any_call": jnp.asarray(s.any_call, jnp.bool_),
        "bids": i8(s.bids),
        "n_bids": i8(s.n_bids),
        "first_cards": s.first_cards & (seat == 0),
        "dealt": dealt,
        "hand": dealt & ~s.played,
        "plays": i8(s.plays),
        "play_seat": i8(s.play_seat),
        "declared": jnp.asarray(s.declared, jnp.bool_),
        "n_played": i8(s.n_played),
        "team_score": jnp.asarray(s.team_score, jnp.int32),
        "pending": jnp.asarray(s.pending, jnp.int32),
        "team_tricks": i8(s.team_tricks),
    }

  def labels(self, s: schnaps.GameState, seat) -> Mapping[str, jax.Array]:
    """The hidden-information labels: the deal.

    Args:
      s: The state.
      seat: Unused (the labels are the same for every seat).

    Returns:
      {"deal_holder": int8[20]}, the seat dealt each card (-1 while undealt).
    """
    del seat
    return {"deal_holder": s.holder.astype(jnp.int8)}


class SchnapsNormalEnv(SchnapsEnv):
  """Schnaps with the Normal game only: pass is the only bid.

  OpenSpiel's schnaps(normal_only=true). The trump choice and the card play
  are schnaps's; the bidding is four forced passes, so there are no flecks
  and no called contracts. Its own env name, so its artifacts never mix with
  full schnaps. No options. The InfoState, actions and sides are schnaps's
  at its default max_flecks (the bid slots keep their size), so schnaps's
  encoders and rule bots read it unchanged.
  """

  name = "schnaps_normal"

  def __init__(self):  # pylint: disable=super-init-not-called
    self.options = {}
    self.max_len = decisions_bound(schnaps.DEFAULT_MAX_FLECKS)
    jax_game_env.JaxGameEnv.__init__(
        self,
        schnaps.Schnaps(
            max_flecks=schnaps.DEFAULT_MAX_FLECKS, normal_only=True
        ),
    )
    assert self.num_players == NUM_PLAYERS and self.num_actions == NUM_ACTIONS

  @property
  def declared_rules_sha(self) -> str:
    """The rules identity: schnaps's RULES with normal_only set."""
    return rules_sha(
        {"max_flecks": schnaps.DEFAULT_MAX_FLECKS, "normal_only": True}
    )


def hide_hidden_cards(
    env: SchnapsEnv, s: schnaps.GameState, seat, key
) -> schnaps.GameState:
  """Privacy hook (check_env and the privacy tests).

  Re-deals what `seat` cannot see: the unplayed cards of the other seats and
  the cards not dealt yet are shuffled among those places, each place
  keeping its count. The Caller's first-cards mark travels with each card,
  so another seat cannot learn which three came first.

  Args:
    env: Unused.
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key of the shuffle.

  Returns:
    The state with the hidden cards re-dealt.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  idx = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  holder = s.holder.astype(jnp.int32)
  hidden = (holder != seat) & ~s.played
  src = jnp.argsort(jnp.where(hidden, idx, NUM_CARDS + idx))
  u = jax.random.uniform(key, (NUM_CARDS,), dtype=jnp.float32)
  dst = jnp.argsort(jnp.where(hidden, u, 2.0 + idx.astype(jnp.float32)))
  return s._replace(
      holder=s.holder.at[dst].set(s.holder[src]),
      first_cards=s.first_cards.at[dst].set(s.first_cards[src]),
  )


def options_check(kw: Mapping[str, Any]) -> dict[str, Any]:
  """The canonical options of an EnvSpec (envs.registry canon).

  Args:
    kw: The options.

  Returns:
    The canonical options.

  Raises:
    TypeError: max_flecks is not an int.
    ValueError: max_flecks is out of range.
  """
  v = kw["max_flecks"]
  if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
    raise TypeError(f"schnaps max_flecks must be an int, got {v!r}")
  if not 0 <= int(v) <= schnaps.MAX_MAX_FLECKS:
    raise ValueError(
        f"schnaps max_flecks must be in 0..{schnaps.MAX_MAX_FLECKS}, got {v}"
    )
  return {"max_flecks": int(v)}
