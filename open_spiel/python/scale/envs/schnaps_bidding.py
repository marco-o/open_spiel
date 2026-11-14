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

"""Schnaps bidding from value estimates, with pluggable bid rules.

A bid is chosen in two steps, so that the second can change without the
first:

1. **Estimates** (`estimate`). For every legal bid, what the bidder's pair
   should expect if it makes that bid and the bidding then ends (everyone
   else passes). A value function (the player's networks) judges the
   counterfactual information state at the first play: `counterfactual`
   applies the bid, then passes, to the public bidding fields of the
   bidder's own InfoState, so only the bidder's information is used.

   | bid                         | judged by                    | value        |
   |-----------------------------|------------------------------|--------------|
   | pass, no call yet           | the Normal game's value      | V            |
   | pass after a call c         | contract c's value           | V * 2^flecks |
   | fleck                       | as pass                      | 2 * pass     |
   | call Schnapser/Lomt/Bauern  | that contract, called by me  | V            |
   | call Jodler / Hermort       | (wins at once)               | +18 / +24    |

   The counterfactual is judged with its flecks cleared and the doubling
   applied afterwards: the contract networks trained without flecks.

2. **A bid rule** (`BidRule`) picks the bid from a `BidContext`: the
   InfoState, the legal mask, the estimates and the set situation
   (`SetContext`). Rules register by name (`register_rule`), so an artifact
   records `{"rule", "rule_kwargs"}` and a new rule (for example one that
   plays for the set score) needs nothing else changed.

   * "greedy": the legal bid of highest value; a call or a fleck must beat
     passing by `margin` points.
   * "pass": always pass (the Normal-only baseline).

The set situation. `set_context(info)` reads an InfoState key "set_need"
(i32[B, 2], points each partnership still needs, seats 0/2 first) when an
env provides it, else marks the context unknown. No env provides it yet;
the rules here ignore it.

Simplifications: the estimate is myopic (it ignores later calls and flecks
by the others), and the contract networks never saw overcalls or flecks in
training, so such states are outside what they learned.
"""

from typing import Any, Callable, Mapping, NamedTuple, Protocol, Union

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as game

NUM_ACTIONS = game.NUM_DISTINCT_ACTIONS
PASS = game.PASS_ACTION
FLECK = game.FLECK_ACTION
#: The contracts a value function judges (index = contract id).
VALUED_CONTRACTS = (game.NORMAL, game.SCHNAPSER, game.LOMT, game.BAUERN)
#: The contracts that win at once when called.
RESOLVED_CONTRACTS = (game.JODLER, game.HERMORT)


def call_action(contract: int) -> int:
  """The action that calls `contract` (Schnapser..Hermort).

  Args:
    contract: the contract id.

  Returns:
    The action id.
  """
  return game.FIRST_CALL_ACTION + contract - 1


class SetContext(NamedTuple):
  """The set situation of each row (module doc).

  Attributes:
    known: bool[B], whether a set is being played.
    our_need: i32[B], points the bidder's partnership still needs.
    their_need: i32[B], points the other partnership still needs.
  """

  known: jax.Array
  our_need: jax.Array
  their_need: jax.Array


class BidEstimates(NamedTuple):
  """What each bid is worth (module doc), per action id; bids only.

  Attributes:
    value: f32[B, 39], expected points for the bidder's pair if it makes this
      bid and the bidding then ends; -inf on actions that are not legal bids.
    stake: f32[B, 39], points won or lost on the resulting called contract
      (points * 2^flecks); 0 for a Normal game.
    p_success: f32[B, 39], the chance the declarer's pair makes the resulting
      contract (from value and stake); NaN for a Normal game.
    my_side_declares: bool[B, 39], whether the bidder's pair is the
      declarer's.
  """

  value: jax.Array
  stake: jax.Array
  p_success: jax.Array
  my_side_declares: jax.Array


class BidContext(NamedTuple):
  """Everything a bid rule sees.

  Attributes:
    info: the bidder's InfoState, leaves [B, ...].
    legal: bool[B, 39].
    estimates: the BidEstimates.
    set: the SetContext.
  """

  info: Mapping[str, jax.Array]
  legal: jax.Array
  estimates: BidEstimates
  set: SetContext


class BidRule(Protocol):
  """Chooses a bid from a BidContext (module doc).

  Attributes:
    name: the registered name.
  """

  name: str

  def choose(self, ctx: BidContext, key: jax.Array) -> jax.Array:
    """i32[B], the chosen bid of each row (a legal bid on bidding rows)."""
    ...

  def kwargs(self) -> dict[str, Any]:
    """The JSON-able kwargs that rebuild the rule."""
    ...


_RULES: dict[str, Callable[..., BidRule]] = {}


def register_rule(name: str, ctor: Callable[..., BidRule]) -> None:
  """Registers a bid rule constructor under `name`.

  Args:
    name: the rule name.
    ctor: ctor(**kwargs) -> BidRule.

  Raises:
    ValueError: the name is taken.
  """
  if name in _RULES:
    raise ValueError(f"bid rule {name!r} is already registered")
  _RULES[name] = ctor


def make_rule(name: str, kwargs: Mapping[str, Any] | None = None) -> BidRule:
  """The bid rule `name` built with `kwargs`.

  Args:
    name: a registered rule name.
    kwargs: its kwargs.

  Returns:
    The rule.

  Raises:
    KeyError: unknown rule.
    ValueError: the rule reports another name.
  """
  if name not in _RULES:
    raise KeyError(f"unknown bid rule {name!r}; registered: {sorted(_RULES)}")
  rule = _RULES[name](**dict(kwargs or {}))
  if rule.name != name:
    raise ValueError(f"bid rule {name!r} built a rule named {rule.name!r}")
  return rule


def registered_rules() -> list[str]:
  """The registered rule names."""
  return sorted(_RULES)


def _bid_mask(legal: jax.Array) -> jax.Array:
  """The legal bids: legal actions among pass, fleck and the calls."""
  acts = jnp.arange(NUM_ACTIONS)
  return legal & (acts >= PASS)


class GreedyRule:
  """The legal bid of highest value; calls and flecks must beat pass by margin.

  Attributes:
    name: "greedy".
    margin: points a call or fleck must add over passing.
  """

  name = "greedy"

  def __init__(self, margin: float = 0.0):
    self.margin = float(margin)

  def kwargs(self) -> dict[str, Any]:
    return {"margin": self.margin}

  def choose(self, ctx: BidContext, key: jax.Array) -> jax.Array:
    del key
    acts = jnp.arange(NUM_ACTIONS)
    score = ctx.estimates.value - jnp.where(acts > PASS, self.margin, 0.0)
    score = jnp.where(_bid_mask(ctx.legal), score, -jnp.inf)
    # Ties go to the lowest action id: pass first.
    return jnp.argmax(score, axis=-1).astype(jnp.int32)


class PassRule:
  """Always pass.

  Attributes:
    name: "pass".
  """

  name = "pass"

  def kwargs(self) -> dict[str, Any]:
    return {}

  def choose(self, ctx: BidContext, key: jax.Array) -> jax.Array:
    del key
    return jnp.full(ctx.legal.shape[:1], PASS, jnp.int32)


register_rule("greedy", GreedyRule)
register_rule("pass", PassRule)


def set_context(info: Mapping[str, jax.Array]) -> SetContext:
  """The rows' set situation (module doc).

  Args:
    info: the InfoState, leaves [B, ...].

  Returns:
    The SetContext; unknown (all zeros) unless info has "set_need".
  """
  seat = jnp.asarray(info["seat"], jnp.int32)
  if "set_need" in info:
    need = jnp.asarray(info["set_need"], jnp.int32)
    us = seat % 2
    rows = jnp.arange(seat.shape[0])
    return SetContext(
        known=jnp.ones(seat.shape, jnp.bool_),
        our_need=need[rows, us],
        their_need=need[rows, 1 - us],
    )
  zeros = jnp.zeros(seat.shape, jnp.int32)
  return SetContext(
      known=jnp.zeros(seat.shape, jnp.bool_), our_need=zeros, their_need=zeros
  )


def apply_bid(info: Mapping[str, jax.Array], a) -> dict[str, jax.Array]:
  """One seat's InfoState after the next bid `a` (unbatched).

  Mirrors jax.games.schnaps.Schnaps._bid_step on the public bidding fields
  (the bidder is seat n_bids % 4). When the bid ends the bidding, the phase
  becomes play, or over for Jodler and Hermort (whose scores are not
  filled in).

  Args:
    info: one seat's InfoState, in the bidding phase.
    a: the bid (32..38).

  Returns:
    The InfoState after the bid.
  """
  a = jnp.asarray(a, jnp.int32)
  i32 = lambda k: jnp.asarray(info[k], jnp.int32)
  n_bids = i32("n_bids")
  p = n_bids % game.NUM_PLAYERS
  bidding = i32("phase") == game.PHASE_BIDDING
  is_pass = a == PASS
  is_fleck = a == FLECK
  is_call = (a >= game.FIRST_CALL_ACTION) & (a < NUM_ACTIONS)
  passes = jnp.where(is_pass, i32("passes") + 1, 0)
  contract = jnp.where(is_call, a - game.FIRST_CALL_ACTION + 1, i32("contract"))
  declarer = jnp.where(is_call, p, i32("declarer"))
  any_call = jnp.asarray(info["any_call"], jnp.bool_) | is_fleck | is_call
  flecks = jnp.where(
      is_fleck, i32("flecks") + 1, jnp.where(is_call, 0, i32("flecks"))
  )
  last_actor = jnp.where(is_fleck | is_call, p, i32("last_actor"))
  bids = jnp.asarray(info["bids"])
  bids = bids.at[n_bids].set((a - PASS).astype(bids.dtype), mode="drop")
  end = passes >= jnp.where(any_call, 3, 4)
  resolved = (contract == game.JODLER) | (contract == game.HERMORT)
  phase = jnp.where(
      end,
      jnp.where(resolved, game.PHASE_OVER, game.PHASE_PLAY),
      game.PHASE_BIDDING,
  )

  def put(k, v):
    return jnp.where(bidding, jnp.asarray(v, info[k].dtype), info[k])

  out = dict(info)
  out.update(
      phase=put("phase", phase),
      contract=put("contract", contract),
      declarer=put("declarer", declarer),
      flecks=put("flecks", flecks),
      passes=put("passes", passes),
      any_call=put("any_call", any_call),
      last_actor=put("last_actor", last_actor),
      bids=jnp.where(bidding, bids, info["bids"]),
      n_bids=put("n_bids", n_bids + 1),
  )
  return out


def counterfactual(info: Mapping[str, jax.Array], a) -> dict[str, jax.Array]:
  """One seat's InfoState after bid `a` and then passes to the end (unbatched).

  Args:
    info: one seat's InfoState, in the bidding phase.
    a: the bid.

  Returns:
    The InfoState when the bidding has ended (play, or over).
  """
  out = apply_bid(info, a)
  for _ in range(game.NUM_PLAYERS):  # at most 3 more passes are needed
    out = apply_bid(out, PASS)  # a no-op once the bidding has ended
  return out


#: value_fn(info_after, contract) -> f32[B]: the expected points of the
#: viewer's pair at the first play of `info_after`, read by the value
#: function of `contract` (a Python int for every row, or i32[B]).
ValueFn = Callable[[Mapping[str, jax.Array], Union[int, jax.Array]], jax.Array]


def _clear_flecks(info):
  out = dict(info)
  out["flecks"] = jnp.zeros_like(info["flecks"])
  return out


def estimate(
    info: Mapping[str, jax.Array], legal: jax.Array, value_fn: ValueFn
) -> BidEstimates:
  """The BidEstimates of every row (module doc).

  Args:
    info: the bidders' InfoStates, leaves [B, ...].
    legal: bool[B, 39].
    value_fn: the value function (ValueFn).

  Returns:
    The BidEstimates.
  """
  seat = jnp.asarray(info["seat"], jnp.int32)
  b = seat.shape[0]
  points = jnp.asarray(game.CONTRACT_POINTS, jnp.float32)
  cf = jax.vmap(counterfactual)

  value = jnp.full((b, NUM_ACTIONS), -jnp.inf, jnp.float32)
  stake = jnp.zeros((b, NUM_ACTIONS), jnp.float32)
  mine = jnp.zeros((b, NUM_ACTIONS), jnp.bool_)

  # Pass (and fleck): the contract standing now.
  after = cf(info, jnp.full((b,), PASS, jnp.int32))
  contract = jnp.clip(jnp.asarray(after["contract"], jnp.int32), 0, 3)
  mult = 2.0 ** jnp.asarray(after["flecks"], jnp.float32)
  v_pass = value_fn(_clear_flecks(after), contract) * mult
  called = contract != game.NORMAL
  s_pass = jnp.where(called, points[contract] * mult, 0.0)
  m_pass = jnp.asarray(after["declarer"], jnp.int32) % 2 == seat % 2
  value = value.at[:, PASS].set(v_pass)
  stake = stake.at[:, PASS].set(s_pass)
  mine = mine.at[:, PASS].set(m_pass)
  value = value.at[:, FLECK].set(2.0 * v_pass)
  stake = stake.at[:, FLECK].set(2.0 * s_pass)
  mine = mine.at[:, FLECK].set(m_pass)

  # Calls judged by their contract's value function.
  for c in VALUED_CONTRACTS[1:]:
    a = call_action(c)
    after_c = cf(info, jnp.full((b,), a, jnp.int32))
    value = value.at[:, a].set(value_fn(after_c, c))
    stake = stake.at[:, a].set(points[c])
    mine = mine.at[:, a].set(True)
  # Calls that win at once.
  for c in RESOLVED_CONTRACTS:
    a = call_action(c)
    value = value.at[:, a].set(points[c])
    stake = stake.at[:, a].set(points[c])
    mine = mine.at[:, a].set(True)

  bids = _bid_mask(legal)
  value = jnp.where(bids, value, -jnp.inf)
  declarer_value = jnp.where(mine, value, -value)
  p = jnp.clip((declarer_value / jnp.maximum(stake, 1e-9) + 1.0) / 2.0, 0, 1)
  p = jnp.where(bids & (stake > 0), p, jnp.nan)
  return BidEstimates(
      value=value, stake=stake, p_success=p, my_side_declares=mine & bids
  )
