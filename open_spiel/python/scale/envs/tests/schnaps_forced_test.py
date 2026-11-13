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

"""envs.schnaps_forced: forced-contract deals, and the hands script."""

import json
import sys

import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import schnaps as G
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import schnaps_forced as sf
from open_spiel.python.scale.envs import schnaps_oracle as oracle
from open_spiel.python.scale.examples import schnaps_forced_hands as hands_script
from open_spiel.python.scale.testing import check_env

# pylint: enable=g-import-not-at-top,wrong-import-position

ENVS = ("schnaps_forced_schnapser", "schnaps_forced_lomt", "schnaps_forced_bauern")
N = 64


@pytest.fixture(scope="module", autouse=True)
def _x64():
  x64.enable_x64()


def _deal(env, n=N, seed=0):
  keys = jax.random.split(jax.random.key(seed), n)
  return jax.jit(jax.vmap(env.init))(keys)


def _row(s, i):
  return jax.tree_util.tree_map(lambda x: x[i], s)


def _random_playout(env, s, seed):
  """Plays one forced deal (unbatched) to the end at random.

  Args:
    env: the env.
    s: a first-decision state.
    seed: the seed.

  Returns:
    (the terminal state, the actions played).
  """
  rng = np.random.default_rng(seed)
  step = jax.jit(env.step)
  legal = jax.jit(env.legal_mask)
  term = jax.jit(env.terminal)
  played = []
  while not bool(term(s)):
    a = int(rng.choice(np.flatnonzero(np.asarray(legal(s)))))
    played.append(a)
    s = step(s, jnp.int32(a), jax.random.key(0))
  return s, played


@pytest.mark.parametrize("name", ENVS)
def test_the_first_decision_is_a_play_of_the_forced_contract(name):
  env = env_registry.make(name)
  contract = sf.CONTRACTS[env.contract_name]
  s = _deal(env)
  b = s.base
  d = np.asarray(s.declarer, np.int32)
  assert (np.asarray(s.stage) == sf.STAGE_PLAY).all()
  assert (np.asarray(b.phase) == G.PHASE_PLAY).all()
  assert (np.asarray(b.contract) == contract).all()
  assert (np.asarray(b.declarer) == d).all()
  assert (np.asarray(b.trump) == np.asarray(s.trump)).all()
  assert (np.asarray(b.flecks) == 0).all()
  assert set(d.tolist()) == {0, 1, 2, 3}
  leader = d if contract == G.LOMT else np.zeros_like(d)
  assert (np.asarray(b.current_player) == leader).all()
  holder = np.asarray(b.holder)
  assert (holder == np.asarray(s.holder)).all()
  for i in range(N):
    assert np.bincount(holder[i], minlength=4).tolist() == [5, 5, 5, 5]
    bids = np.asarray(b.bids[i]).tolist()
    want = [0] * d[i] + [contract + 1] + [0, 0, 0]
    assert bids[: len(want)] == want and int(b.n_bids[i]) == len(want)
    first = np.flatnonzero(np.asarray(b.first_cards[i]))
    assert len(first) == 3 and (holder[i, first] == 0).all()
    trump = int(s.trump[i])
    trumps = [c for c in np.flatnonzero(holder[i] == 0) if c // 5 == trump]
    assert sum(c // 5 == trump for c in first) == min(len(trumps), 3)


@pytest.mark.parametrize("name", ENVS)
def test_forced_deals_are_schnaps_deals_in_pyspiel(name):
  pyspiel = pytest.importorskip("pyspiel")
  env = env_registry.make(name)
  game = pyspiel.load_game(f"schnaps(max_flecks={sf.MAX_FLECKS})")
  base_game = G.Schnaps(max_flecks=sf.MAX_FLECKS)
  s = _deal(env, n=6, seed=3)
  for i in range(6):
    first = _row(s, i)
    script = [int(a) for a in np.asarray(env.game.script(first)) if a >= 0]
    end, played = _random_playout(env, first, seed=i)
    st = game.new_initial_state()
    for a in script + played:
      st.apply_action(a)
    assert st.is_terminal()
    assert np.allclose(st.returns(), np.asarray(env.returns(end)))
    os_base = base_game.state_from_os(st)
    for x, y in zip(
        jax.tree_util.tree_leaves(os_base), jax.tree_util.tree_leaves(end.base)
    ):
      assert np.array_equal(np.asarray(x), np.asarray(y))


def _hands_file(tmp_path, contract_name, weight):
  contract = sf.CONTRACTS[contract_name]
  classes, _ = sf.hand_classes(sf.trump_marked_of(contract))
  data = {
      "schema": sf.HANDS_SCHEMA,
      "contracts": {
          contract_name: {
              "classes": [list(k) for k in classes],
              "weight": weight(len(classes)),
          }
      },
  }
  path = tmp_path / "hands.json"
  path.write_text(json.dumps(data))
  return str(path), classes


def _declarer_classes(env, s, marked):
  holder = np.asarray(s.holder)
  d = np.asarray(s.declarer)
  t = np.asarray(s.trump)
  masks = sf.suit_masks(holder == d[:, None])
  classes, _ = sf.hand_classes(marked)
  index = {k: i for i, k in enumerate(classes)}
  return np.array(
      [index[sf.class_key(m, int(tt), marked)] for m, tt in zip(masks, t)]
  ), d


def test_lomt_declarer_hands_follow_the_weights(tmp_path):
  classes, _ = sf.hand_classes(False)
  a, b = classes.index((31, 0, 0, 0)), classes.index((28, 17, 0, 0))

  def weight(n):
    w = [0.0] * n
    w[a], w[b] = 1.0, 3.0
    return w

  path, _ = _hands_file(tmp_path, "lomt", weight)
  env = sf.SchnapsForcedLomtEnv(hands_path=path)
  ids, _ = _declarer_classes(env, _deal(env, n=4000, seed=1), False)
  assert set(ids.tolist()) <= {a, b}
  counts = sf.class_counts(False)
  want = counts[b] * 3.0 / (counts[a] * 1.0 + counts[b] * 3.0)
  got = float((ids == b).mean())
  assert abs(got - want) < 4 * np.sqrt(want * (1 - want) / len(ids)) + 1e-9


def test_trump_marked_weights_follow_the_seat(tmp_path):
  classes, _ = sf.hand_classes(True)
  a = classes.index((31, 0, 0, 0))  # the whole trump suit
  b = classes.index((0, 31, 0, 0))  # a whole side suit

  def weight(n):
    w = [[0.0] * 4 for _ in range(n)]
    w[a] = [1.0, 0.0, 1.0, 0.0]
    w[b] = [0.0, 1.0, 0.0, 1.0]
    return w

  path, _ = _hands_file(tmp_path, "bauern", weight)
  env = sf.SchnapsForcedBauernEnv(hands_path=path)
  ids, d = _declarer_classes(env, _deal(env, n=256, seed=2), True)
  assert (ids[d % 2 == 0] == a).all() and (ids[d % 2 == 1] == b).all()


@pytest.mark.parametrize("name", ENVS)
def test_check_env_passes_with_privacy(name):
  env = env_registry.make(name)
  check_env.check_env(env, hide_fn=sf.hide_hidden_cards, num_playouts=16)


def test_the_shipped_hands_file_covers_every_contract():
  data, sha = sf.load_hands()
  assert len(sha) == 64
  for name, contract in sf.CONTRACTS.items():
    entry = data["contracts"][name]
    w = sf.hand_weights(entry, contract)
    assert w.shape == (4, 4, sf.NUM_HANDS) and (w >= 0).all()
    p = np.asarray(entry["p"])
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize(
    "contract, key, seat, want",
    [
        (oracle.LOMT, (31, 0, 0, 0), 0, 1.0),  # a whole suit, no trump
        (oracle.BAUERN, (31, 0, 0, 0), 3, 1.0),  # every trump
        (oracle.BAUERN, (0, 15, 1, 0), 2, 0.0),  # no trump, no As
    ],
)
def test_success_rate_of_obvious_hands(contract, key, seat, want):
  assert hands_script.success_rate((contract, 0, key, seat, 6, 0)) == want


def test_the_hands_script_on_a_few_classes():
  entry = hands_script.contract_entry(
      "lomt",
      samples=4,
      seed=0,
      lo=0.02,
      hi=0.15,
      weak_share=0.15,
      max_classes=3,
      pool=None,
  )
  classes, _ = sf.hand_classes(False)
  assert entry["classes"] == [list(k) for k in classes]
  assert entry["seats"] == [0] and len(entry["p"]) == len(classes)
  assert all(p == 0 for p in entry["p"][3:])
  again = hands_script.contract_entry(
      "lomt",
      samples=4,
      seed=0,
      lo=0.02,
      hi=0.15,
      weak_share=0.15,
      max_classes=3,
      pool=None,
  )
  assert again == entry


def test_class_weights_give_the_weak_share():
  p = np.array([0.0, 0.05, 0.1, 0.5, 0.9])
  counts = np.array([10, 20, 20, 30, 10])
  w = sf.class_weights(p, counts, lo=0.02, hi=0.15, weak_share=0.2)
  assert w[0] == 0 and w[3] == w[4] == 1
  mass = counts * w
  assert np.isclose(mass[1:3].sum() / mass.sum(), 0.2)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
