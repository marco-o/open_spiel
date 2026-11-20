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

"""learners.pg aux heads: the targets, the per-row losses and the config."""

import math
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.learners import pg


def test_hand_returns_sum_to_the_hand_end_and_flag_open_hands():
  #            hand A ends at t=2 | hand B runs past the segment
  reward = jnp.array([[0.0, 0.0, 2.0, 0.0, 1.0]], jnp.float32)
  done = jnp.array([[False, False, True, False, False]])
  g, ends = pg.hand_returns(reward, done)
  np.testing.assert_allclose(np.asarray(g), [[2.0, 2.0, 2.0, 1.0, 1.0]])
  assert np.asarray(ends).tolist() == [[True, True, True, False, False]]


def test_hand_returns_restart_after_each_hand():
  reward = jnp.array([[-1.0, 3.0, 0.0, -2.0]], jnp.float32)
  done = jnp.array([[True, True, False, True]])
  g, ends = pg.hand_returns(reward, done)
  np.testing.assert_allclose(np.asarray(g), [[-1.0, 3.0, -2.0, -2.0]])
  assert np.asarray(ends).all()


def test_token_class_loss_skips_unlabelled_tokens():
  # 1 row, 3 tokens, 2 classes; token 2 has no label
  pred = jnp.array([[[0.0, 0.0], [5.0, -5.0], [9.0, -9.0]]], jnp.float32)
  label = jnp.array([[1, 0, -1]], jnp.int8)
  loss, acc, has = pg.aux_rows_loss("token_class", pred, label)
  want = (math.log(2.0) + float(jnp.log1p(jnp.exp(-10.0)))) / 2
  np.testing.assert_allclose(np.asarray(loss), [want], rtol=1e-5)
  np.testing.assert_allclose(np.asarray(acc), [0.5])  # tie argmax 0 != 1
  assert np.asarray(has).tolist() == [True]
  _, _, none = pg.aux_rows_loss(
      "token_class", pred, jnp.full((1, 3), -1, jnp.int8)
  )
  assert np.asarray(none).tolist() == [False]
  assert np.asarray(pg.aux_label_rows("token_class", label)).tolist() == [True]


def test_final_score_loss_shifts_and_clips_the_score():
  pred = jnp.zeros((3, 7), jnp.float32).at[0, 6].set(10.0)
  label = jnp.array([3, 9, -3], jnp.int32)  # 9 clips to +3
  loss, acc, has = pg.aux_rows_loss("final_score", pred, label, score_min=-3)
  assert float(loss[0]) < 1e-3 and float(acc[0]) == 1.0
  np.testing.assert_allclose(float(loss[1]), math.log(7.0), rtol=1e-5)
  assert np.asarray(has).all()


def test_aux_target_and_config_validation():
  with pytest.raises(ValueError):
    pg.AuxTarget("x", "nope")
  with pytest.raises(ValueError):
    pg.AuxTarget("x", "token_class")  # needs a label
  with pytest.raises(ValueError):
    pg.AuxTarget("x", "final_score", label="y")
  t = pg.AuxTarget("x", "final_score", score_min=-3)
  with pytest.raises(ValueError):
    pg.PGConfig(loss="ppo", anneal_lr=False, aux=(t, t))
  assert pg.PGConfig(loss="ppo", anneal_lr=False, aux=(t,)).aux == (t,)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
