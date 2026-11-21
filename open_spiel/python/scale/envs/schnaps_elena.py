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

"""Elena, a schnaps rule bot, and the evaluation against her.

Elena plays every schnaps decision by fixed rules (Elena.md), reading only
her own seat's InfoState:

Trump (the Caller, three cards seen): the suit whose cards add up to the
most points; ties at random.

Bidding, the first rule that applies and is legal:
  1. Hermort, else Jodler, whenever legal (she holds a whole suit).
  2. Bauern: at least 3 safe trump takes and 4 safe takes in all.
  3. Lomt: at least 4 safe takes.
  4. Schnapser: 3 safe trump takes, in the Caller's partnership (seats 0, 2).
  5. Fleck the Normal game: 3 safe trump takes, in the other partnership.
  6. Fleck an opposing Bauern: 3 or more trumps, or 2 trumps one of which is
     the trump Zehn.
  7. Otherwise pass.
A safe take is a card all of whose higher cards of its suit she holds too
(the As always is one): per suit, the unbroken run down from the As.

Play:
  * leading: a suit in which every higher card than her best one has been
    played (her best card of it is the boss), preferring trump, then at
    random; she leads that best card. With no such suit, her lowest card
    (non-trump first on a tie, then at random). Leading a Koenig or Ober
    whose partner card she holds, she declares the 20 or 40.
  * following: her lowest legal card (non-trump first on a tie, then at
    random).

Lomt is played without trump, so there "trump" preferences do not apply.

  pol = policies.registry.build("schnaps.elena", {})
  ev = eval.registry.build("schnaps.elena", env)          # env: a SchnapsEnv
  res = ev.vs_reference(entrant, n_deals=512, key=jax.random.key(0))

The evaluation plays partnership against partnership on seat-swapped
duplicate deals; the payoff is points per deal from the entrant's side.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as game
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import schnaps
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import registry as eval_registry
from open_spiel.python.scale.policies import fixed_action
from open_spiel.python.scale.policies import registry

ARCH = "schnaps.elena"
EVAL_NAME = "schnaps.elena"
#: The same evaluation on schnaps_normal (each game names its own evals).
NORMAL_EVAL_NAME = "schnaps_normal.elena"
ENTRANT_NAME = "elena"

_N_SUITS, _N_RANKS = game.NUM_SUITS, game.NUM_RANKS
_POINTS = jnp.asarray(game.RANK_POINTS, jnp.int32)
_CARD_SUIT = jnp.arange(game.NUM_CARDS) // _N_RANKS
_CARD_RANK = jnp.arange(game.NUM_CARDS) % _N_RANKS


def _call(contract: int) -> int:
  """The action that calls `contract`.

  Args:
    contract: SCHNAPSER .. HERMORT.

  Returns:
    The action.
  """
  return game.FIRST_CALL_ACTION + contract - 1


def safe_takes(hand) -> jax.Array:
  """i32[4]: each suit's safe takes, the unbroken run down from its As.

  Args:
    hand: bool[20].

  Returns:
    i32[4].
  """
  by_suit = hand.reshape(_N_SUITS, _N_RANKS)[:, ::-1].astype(jnp.int32)
  return jnp.cumprod(by_suit, axis=1).sum(axis=1)


def _pick(score, key) -> jax.Array:
  """The index of the highest score, ties broken uniformly at random.

  Args:
    score: f32[n]; candidates scored -inf are never picked while any other is.
    key: a PRNG key.

  Returns:
    i32 index.
  """
  noise = jax.random.uniform(key, score.shape, jnp.float32, 0.0, 0.5)
  return jnp.argmax(score + noise).astype(jnp.int32)


def trump_choice(info, key) -> jax.Array:
  """The Caller's trump action: the suit of the most card points.

  Args:
    info: one seat's InfoState (unbatched).
    key: a PRNG key.

  Returns:
    i32 action, FIRST_TRUMP_ACTION + suit.
  """
  by_suit = info["hand"].reshape(_N_SUITS, _N_RANKS).astype(jnp.int32)
  sums = (by_suit * _POINTS).sum(axis=1).astype(jnp.float32)
  return game.FIRST_TRUMP_ACTION + _pick(sums, key)


def bid(info, legal) -> jax.Array:
  """Elena's bid (module doc).

  Args:
    info: one seat's InfoState (unbatched).
    legal: bool[39].

  Returns:
    i32 action.
  """
  hand = info["hand"]
  trump = info["trump"].astype(jnp.int32)
  safe = safe_takes(hand)
  total_safe = safe.sum()
  trump_safe = safe[trump]
  trump_cards = hand & (_CARD_SUIT == trump)
  n_trump = trump_cards.sum()
  has_trump_zehn = trump_cards[trump * _N_RANKS + game.ZEHN]
  callers_side = info["seat"].astype(jnp.int32) % 2 == 0
  contract = info["contract"].astype(jnp.int32)
  fleck = game.FLECK_ACTION
  rules = (
      (legal[_call(game.HERMORT)], _call(game.HERMORT)),
      (legal[_call(game.JODLER)], _call(game.JODLER)),
      (
          (trump_safe >= 3) & (total_safe >= 4) & legal[_call(game.BAUERN)],
          _call(game.BAUERN),
      ),
      ((total_safe >= 4) & legal[_call(game.LOMT)], _call(game.LOMT)),
      (
          (trump_safe >= 3) & callers_side & legal[_call(game.SCHNAPSER)],
          _call(game.SCHNAPSER),
      ),
      (
          (trump_safe >= 3)
          & ~callers_side
          & (contract == game.NORMAL)
          & legal[fleck],
          fleck,
      ),
      (
          ((n_trump >= 3) | ((n_trump == 2) & has_trump_zehn))
          & (contract == game.BAUERN)
          & legal[fleck],
          fleck,
      ),
  )
  action = jnp.int32(game.PASS_ACTION)
  for cond, a in reversed(rules):
    action = jnp.where(cond, jnp.int32(a), action)
  return action


def _lowest(cards, trump, key) -> jax.Array:
  """The lowest card of `cards`: non-trump first on a tie, then at random.

  Args:
    cards: bool[20], not all False.
    trump: the trump suit in play (-1 in Lomt).
    key: a PRNG key.

  Returns:
    i32 card.
  """
  cost = _CARD_RANK * 2 + (_CARD_SUIT == trump)
  score = jnp.where(cards, -cost.astype(jnp.float32), -jnp.inf)
  return _pick(score, key)


def play(info, legal, key) -> jax.Array:
  """Elena's lead or follow (module doc).

  Args:
    info: one seat's InfoState (unbatched).
    legal: bool[39].
    key: a PRNG key.

  Returns:
    i32 action: a card, or the declaration that leads it.
  """
  k_suit, k_low, k_follow = jax.random.split(key, 3)
  hand = info["hand"]
  plays = info["plays"].astype(jnp.int32)
  played = (
      jnp.zeros((game.NUM_CARDS,), jnp.bool_)
      .at[jnp.where(plays >= 0, plays, game.NUM_CARDS)]
      .set(True, mode="drop")
  )
  trump_on = info["contract"].astype(jnp.int32) != game.LOMT
  trump = jnp.where(trump_on, info["trump"].astype(jnp.int32), -1)
  by_suit = hand.reshape(_N_SUITS, _N_RANKS)
  # Each suit's best card in hand, and whether every higher card is played.
  ranks = jnp.arange(_N_RANKS)
  best = jnp.max(jnp.where(by_suit, ranks, -1), axis=1)
  gone = played.reshape(_N_SUITS, _N_RANKS)
  boss = (best >= 0) & jnp.all(gone | (ranks[None, :] <= best[:, None]), axis=1)
  suits = jnp.arange(_N_SUITS)
  suit_score = jnp.where(boss, (suits == trump).astype(jnp.float32), -jnp.inf)
  boss_suit = _pick(suit_score, k_suit)
  boss_card = boss_suit * _N_RANKS + best[boss_suit]
  lead = jnp.where(boss.any(), boss_card, _lowest(hand, trump, k_low))
  # Declare a Koenig or Ober led with its partner card in hand.
  lead_rank = lead % _N_RANKS
  decl = (
      game.FIRST_DECLARE_ACTION
      + 2 * (lead // _N_RANKS)
      + jnp.where(lead_rank == game.KOENIG, 0, 1)
  )
  is_pair_card = (lead_rank == game.KOENIG) | (lead_rank == game.OBER)
  can_decl = is_pair_card & legal[
      jnp.clip(decl, 0, game.NUM_DISTINCT_ACTIONS - 1)
  ]
  lead_action = jnp.where(can_decl, decl, lead)
  follow = _lowest(legal[: game.NUM_CARDS], trump, k_follow)
  leading = info["n_played"].astype(jnp.int32) % game.NUM_PLAYERS == 0
  return jnp.where(leading, lead_action, follow).astype(jnp.int32)


def rule_action(info, legal, key) -> jax.Array:
  """The action Elena's rules pick at one decision (unbatched).

  Args:
    info: one seat's InfoState (unbatched).
    legal: bool[39], the seat's legal actions.
    key: a PRNG key.

  Returns:
    i32 action.
  """
  k_trump, k_play = jax.random.split(key)
  phase = info["phase"].astype(jnp.int32)
  return jnp.where(
      phase == game.PHASE_CHOOSE_TRUMP,
      trump_choice(info, k_trump),
      jnp.where(
          phase == game.PHASE_BIDDING, bid(info, legal), play(info, legal, k_play)
      ),
  ).astype(jnp.int32)


def action(info, legal, key) -> jax.Array:
  """Elena's action at one decision (unbatched).

  A rule that picks an illegal action (none should) falls back to the lowest
  legal action.

  Args:
    info: one seat's InfoState (unbatched).
    legal: bool[39], the seat's legal actions.
    key: a PRNG key.

  Returns:
    i32 action.
  """
  a = rule_action(info, legal, key)
  ok = legal[jnp.clip(a, 0, game.NUM_DISTINCT_ACTIONS - 1)]
  return jnp.where(ok, a, jnp.argmax(legal).astype(jnp.int32))


def actions(info, legal, key) -> jax.Array:
  """i32[B]: Elena's action on each row.

  Args:
    info: a batched InfoState.
    legal: bool[B, 39].
    key: a PRNG key (split per row).

  Returns:
    i32[B].
  """
  keys = jax.random.split(key, legal.shape[0])
  return jax.vmap(action)(info, legal, keys)


class ElenaPolicy(types.Policy):
  """Stateless; params {}; carry ().

  Attributes:
    spec: The policy's PolicySpec.
  """

  ARCH = ARCH
  decider = "scripted"

  def __init__(self):
    self.spec = types.PolicySpec(
        game=schnaps.SchnapsEnv.name,
        obs_format="none",
        num_actions=schnaps.NUM_ACTIONS,
        num_players=schnaps.NUM_PLAYERS,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """Empty features (the policy reads the InfoState directly).

    Args:
      info: the InfoState.

    Returns:
      f32[B, 0].
    """
    return fixed_action.empty_features(info)

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """Elena's action, one-hot, on every row with a legal action.

    Args:
      params: unused.
      carry: unused.
      step: the SeatStep; its info is the batched InfoState.
      key: the evaluate key.

    Returns:
      The PolicyOutput.
    """
    del params, carry
    legal = step.legal_mask
    return fixed_action.one_hot_output(
        actions(step.info, legal, key),
        legal.any(axis=-1),
        schnaps.NUM_ACTIONS,
        schnaps.NUM_PLAYERS,
    )


def entrant() -> match.Entrant:
  """Elena as an entrant (she plays her one action: ARGMAX).

  Returns:
    The Entrant.
  """
  return match.Entrant(ENTRANT_NAME, ElenaPolicy(), {}, types.ARGMAX)


def evaluator(env, mesh=None, *, name: str = EVAL_NAME) -> match.Evaluator:
  """The Evaluator against Elena on `env`, a SchnapsEnv.

  Args:
    env: the SchnapsEnv.
    mesh: the device mesh, or None.
    name: the evaluation name.

  Returns:
    The Evaluator.

  Raises:
    TypeError: env is not a SchnapsEnv.
  """
  if not isinstance(env, schnaps.SchnapsEnv):
    raise TypeError(
        f"the Elena evaluation plays a SchnapsEnv, got {type(env).__name__}"
    )
  return match.Evaluator(
      env,
      match.EvaluatorConfig(name=name, reference=entrant, max_rows=1024),
      mesh=mesh,
  )


registry.register(ARCH, ElenaPolicy)
eval_registry.register(EVAL_NAME, evaluator)


def normal_evaluator(env, mesh=None) -> match.Evaluator:
  """The Elena evaluation on schnaps_normal (NORMAL_EVAL_NAME).

  Args:
    env: the SchnapsNormalEnv.
    mesh: the device mesh, or None.

  Returns:
    The Evaluator.
  """
  return evaluator(env, mesh, name=NORMAL_EVAL_NAME)


eval_registry.register(NORMAL_EVAL_NAME, normal_evaluator)

#: The Elena evaluations on the forced-contract envs (envs.schnaps_forced),
#: by contract: the same duplicate deals, under each env's own name.
FORCED_EVAL_NAMES = {
    c: f"schnaps_forced_{c}.elena" for c in ("schnapser", "lomt", "bauern")
}


def forced_evaluator(name: str):
  """The Elena evaluation constructor named `name` (FORCED_EVAL_NAMES).

  Args:
    name: the evaluation name.

  Returns:
    fn(env, mesh=None) -> Evaluator.
  """

  def make(env, mesh=None) -> match.Evaluator:
    return evaluator(env, mesh, name=name)

  return make


for _forced_name in FORCED_EVAL_NAMES.values():
  eval_registry.register(_forced_name, forced_evaluator(_forced_name))

#: The Elena evaluation on the mixed env (envs.schnaps_mixed): Normal and
#: forced deals in its mix, the same duplicate deals.
MIXED_EVAL_NAME = "schnaps_mixed.elena"
eval_registry.register(MIXED_EVAL_NAME, forced_evaluator(MIXED_EVAL_NAME))
