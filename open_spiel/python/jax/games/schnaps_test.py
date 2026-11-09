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

"""Schnaps known answers, and move for move against OpenSpiel's schnaps.

Random deals rarely give a player a whole suit, so besides the seeded random
playouts the game is checked from a sorted deal (seat 0 holds Herz, seat 1
Caro, seat 2 Pik, seat 3 Kreuz), under each trump, where Jodler and Hermort
are callable. The information-state tensor, this package's own (OpenSpiel's
schnaps has none), is checked to carry exactly what OpenSpiel's
information-state string carries.
"""

import random

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import openspiel_equivalence as equivalence
from open_spiel.python.jax.games import registry
from open_spiel.python.jax.games import schnaps

try:
  import pyspiel  # pytype: disable=import-error

  _HAS_SCHNAPS = "schnaps" in pyspiel.registered_names()
except ImportError:
  _HAS_SCHNAPS = False

_needs_schnaps = absltest.skipUnless(
    _HAS_SCHNAPS, "the installed pyspiel has no schnaps"
)


def _config(game_str: str, playouts: int) -> equivalence.GameConfig:
  """A harness config for `game_str`, its tensors left unchecked.

  Args:
    game_str: The game string.
    playouts: The number of seeded random playouts.

  Returns:
    The config.
  """
  max_flecks = registry.load_game(game_str).max_flecks
  return equivalence.GameConfig(
      game_str,
      playouts=playouts,
      max_steps=schnaps.max_game_length(max_flecks),
      check_observation=False,
      check_information_state=False,
  )


def _sorted_deal(trump: int) -> list[int]:
  """The chance outcomes and trump choice of the sorted deal.

  Cards are dealt in id order: three to seat 0, then (after the trump
  choice) two more to seat 0 and five each to seats 1, 2 and 3.

  Args:
    trump: The suit seat 0 chooses.

  Returns:
    The 21 actions.
  """
  return [0, 1, 2, schnaps.FIRST_TRUMP_ACTION + trump, *range(3, 20)]


def _compiled(game: game_lib.Game) -> equivalence.CompiledFns:
  """The harness's jitted functions of `game`.

  Args:
    game: The JAX game.

  Returns:
    The compiled functions.
  """
  return equivalence.CompiledFns(
      legal_actions_mask=jax.jit(game.legal_actions_mask),
      returns=jax.jit(game.returns),
      observation_tensor=jax.jit(game.observation_tensor),
      information_state_tensor=jax.jit(game.information_state_tensor),
  )


def _play(game: schnaps.Schnaps, actions: list[int]) -> schnaps.GameState:
  """The state after `actions` from the start.

  Args:
    game: The game.
    actions: The actions, chance outcomes included.

  Returns:
    The state.
  """
  apply = jax.jit(game.apply_action)
  state = game.new_initial_state()
  for action in actions:
    state = apply(state, jnp.int32(action))
  return state


class SchnapsTest(absltest.TestCase):

  def test_parameters(self):
    game = registry.load_game("schnaps(max_flecks=3)")
    self.assertEqual(game.max_flecks, 3)
    self.assertEqual(game.num_actions, 39)
    self.assertEqual(
        game.obs_shape, (schnaps.info_state_size(3),)
    )
    with self.assertRaises(ValueError):
      schnaps.Schnaps(max_flecks=7)
    with self.assertRaises(TypeError):
      schnaps.Schnaps(max_flecks=True)

  def test_hermort_wins_at_once(self):
    game = schnaps.Schnaps()
    state = _play(game, _sorted_deal(trump=0))
    mask = np.asarray(game.legal_actions_mask(state))
    hermort = schnaps.FIRST_CALL_ACTION + schnaps.HERMORT - 1
    jodler = schnaps.FIRST_CALL_ACTION + schnaps.JODLER - 1
    self.assertTrue(mask[hermort])
    self.assertFalse(mask[jodler])  # seat 0 holds no whole non-trump suit
    state = _play(
        game, _sorted_deal(0) + [hermort] + [schnaps.PASS_ACTION] * 3
    )
    self.assertTrue(bool(game.is_terminal(state)))
    np.testing.assert_array_equal(game.returns(state), [24, -24, 24, -24])

  def test_flecks_double_the_score(self):
    game = schnaps.Schnaps()
    hermort = schnaps.FIRST_CALL_ACTION + schnaps.HERMORT - 1
    fleck = schnaps.FLECK_ACTION
    passes = [schnaps.PASS_ACTION] * 3
    # Seat 0 calls Hermort, seat 1 flecks, seat 2 flecks back.
    state = _play(
        game, _sorted_deal(0) + [hermort, fleck, fleck] + passes
    )
    np.testing.assert_array_equal(game.returns(state), [96, -96, 96, -96])

  def test_declaring_40_counts_after_a_trick(self):
    game = schnaps.Schnaps()
    passes = [schnaps.PASS_ACTION] * 4
    # Seat 0 holds all of Herz (trump) and declares 40 leading the Koenig.
    state = _play(game, _sorted_deal(0) + passes + [20])
    np.testing.assert_array_equal(state.pending, [40, 0])
    np.testing.assert_array_equal(state.team_score, [0, 0])
    # Seats 1..3 cannot follow or trump; seat 0 takes the trick.
    for _ in range(3):
      mask = np.asarray(game.legal_actions_mask(state))
      state = game.apply_action(state, jnp.int32(np.flatnonzero(mask)[0]))
    np.testing.assert_array_equal(state.pending, [0, 0])
    self.assertEqual(int(state.team_score[0]), 40 + 4 + 2 + 2 + 2)
    self.assertEqual(int(state.current_player), 0)

  def test_batched_random_deals_end(self):
    game = schnaps.Schnaps()
    batch = 64

    def random_action(state, key):
      chance = game.chance_outcomes(state)
      legal = game.legal_actions_mask(state)
      logits = jnp.where(
          game.is_chance_node(state),
          jnp.log(jnp.pad(chance, (0, game.num_actions - 20))),
          jnp.where(legal, 0.0, -jnp.inf),
      )
      return jax.random.categorical(key, logits)

    @jax.jit
    def step(states, key):
      keys = jax.random.split(key, batch)
      actions = jax.vmap(random_action)(states, keys)
      return jax.vmap(game.apply_action)(states, actions)

    states = jax.vmap(lambda _: game.new_initial_state())(jnp.arange(batch))
    key = jax.random.PRNGKey(0)
    for _ in range(game.max_game_length):
      key, sub = jax.random.split(key)
      states = step(states, sub)
    self.assertTrue(bool(jnp.all(states.is_terminal)))
    returns = np.asarray(jax.vmap(game.returns)(states))
    np.testing.assert_array_equal(returns.sum(axis=1), 0)
    self.assertTrue(np.all(np.abs(returns) > 0))


@_needs_schnaps
class SchnapsEquivalenceTest(absltest.TestCase):

  def test_random_playouts(self):
    equivalence.run_playouts(
        self,
        [
            _config("schnaps", 300),
            _config("schnaps(max_flecks=0)", 100),
            _config("schnaps(max_flecks=3)", 100),
        ],
    )

  def test_sorted_deals(self):
    config = _config("schnaps", 1)
    game = registry.load_game("schnaps")
    os_game = pyspiel.load_game("schnaps")
    apply = jax.jit(game.apply_action)
    fns = _compiled(game)
    playout = 0
    for trump in range(schnaps.NUM_SUITS):
      for seed in range(25):
        rng = random.Random(seed)
        os_state = os_game.new_initial_state()
        state = game.new_initial_state()
        prefix = _sorted_deal(trump)
        for step in range(config.max_steps + 1):
          legal = equivalence.compare_state(
              game, os_game, os_state, state, fns, config, playout, step
          )
          if os_state.is_terminal():
            break
          action = prefix[step] if step < len(prefix) else rng.choice(legal)
          os_state.apply_action(action)
          state = apply(state, jnp.int32(action))
        self.assertTrue(os_state.is_terminal())
        playout += 1

  def test_tensor_carries_the_information_state_string(self):
    game = registry.load_game("schnaps")
    os_game = pyspiel.load_game("schnaps")
    tensor = jax.jit(game.information_state_tensor)
    apply = jax.jit(game.apply_action)
    by_string = {}
    by_tensor = {}
    for seed in range(40):
      rng = random.Random(seed)
      os_state = os_game.new_initial_state()
      state = game.new_initial_state()
      while not os_state.is_terminal():
        if not os_state.is_chance_node():
          for player in range(schnaps.NUM_PLAYERS):
            text = os_state.information_state_string(player)
            key = np.asarray(tensor(state, jnp.int32(player))).tobytes()
            self.assertEqual(by_string.setdefault(text, key), key, text)
            self.assertEqual(by_tensor.setdefault(key, text), text, text)
        action = rng.choice(os_state.legal_actions())
        os_state.apply_action(action)
        state = apply(state, jnp.int32(action))
    self.assertGreater(len(by_string), 1000)

  def test_tensor_from_os_matches(self):
    game = registry.load_game("schnaps")
    os_state = pyspiel.load_game("schnaps").new_initial_state()
    rng = random.Random(3)
    for _ in range(35):
      if os_state.is_terminal():
        break
      os_state.apply_action(rng.choice(os_state.legal_actions()))
    state = _play(game, os_state.history())
    for player in range(schnaps.NUM_PLAYERS):
      np.testing.assert_array_equal(
          game.information_state_from_os(os_state, player),
          game.information_state_tensor(state, jnp.int32(player)),
      )


if __name__ == "__main__":
  absltest.main()
