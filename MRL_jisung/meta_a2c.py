"""Meta Asynchronous Actor-Critic Agents (A2C) Agent.

Author: Ji-Sung Kim (Daw Lab)
Email:  hello (at) jisungkim.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time

from absl import logging

import numpy as np
from scipy import signal
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.nn as nn

from .base import Base
from .utils import recursively_update_dict
from .utils import simulate_rollouts
from .utils import makedirs
from .utils import metrics_from_rollouts

DIM_VALUE = 1
DIM_POLICY = 2
EARLY_STOPPING_TEMP_DIR = 'tmp_es'
LOG_OFFSET = 1e-7  # prevent log(0) errors

SAFETY_CHECKS = False
EPISLON = 1e-2


class A2C(Base):
  """TODO"""

  def __init__(self, action_space, sess=None, writer_dir=None,
               params=None, scope=None):
    """Initialize agent."""
    super(A2C, self).__init__(action_space)

    self.params = self.default_params()
    if params:
      recursively_update_dict(self.params, params)

    self.sess = sess if sess is not None else tf.Session()
    self.scope = scope if scope is not None else 'a2c'

    optimizer_params = self.params['optimizer_params']
    if self.params['optimizer'].lower() == 'rmsprop':
      self.optimizer = tf.train.RMSPropOptimizer(**optimizer_params)
    elif self.params['optimizer'].lower() == 'adam':
      self.optimizer = tf.train.AdamOptimizer(**optimizer_params)

    self.network = TrainableArchitecture(
        self.sess, self.optimizer, params=self.params, scope=self.scope)

    self.sess.run(tf.global_variables_initializer())

    self.saver = tf.train.Saver(var_list=self.network.variables)
    if writer_dir is not None:
      self.writer = tf.summary.FileWriter(
          writer_dir, self.sess.graph, flush_secs=10)
    else:
      self.writer = None

    self.reset()

  @staticmethod
  def default_params():  # TODO(jisungk): TUNE
    """Default parameters."""
    return {
        'use_softmax_sampling': False,
        'learning': {
            'gamma': 0.9,
            'value_coef': 0.05,
            'entropy_coef': 0.05,
            'patience': 25,  # set to -1 to disable early stopping
        },
        'architecture_params': Architecture.default_params(),
        'optimizer': 'adam',
        'optimizer_params': {
            'learning_rate': 7e-4,
        },
    }

  def reset(self):
    """Reset the agent."""
    self.prev_action = 0
    self.prev_state = None

  def save(self, path):
    self.saver.save(self.sess, path)

  def load(self, path):
    self.saver.restore(self.sess, path)

  def act(self, observation, prev_reward, done, verbose=False):
    """Request action from agent."""
    params = self.params

    x = np.array([prev_reward, 1 - self.prev_action,
                  self.prev_action]).reshape(1, 1, -1)

    # TODO(jisungk): implement detaching for pytorch variables
    proba, value, lstm_outputs, hidden_state = self.network.forward(
        prev_reward, self.prev_action, self.prev_state)
    proba = np.squeeze(proba)
    value = np.squeeze(value)

    if params['use_softmax_sampling']:
      action = np.random.choice(np.arange(np.size(proba)), p=proba)
    else:
      # TODO(jisungk) this is always biased, check if bias results from this
      action = np.argmax(proba, axis=-1)

    self.prev_action = action
    self.prev_state = hidden_state

    if verbose:
      return {'action': action, 'proba': proba, 'value': value,
              'hidden_state': hidden_state, 'lstm_outputs': lstm_outputs}
    else:
      return action

  def train(self, env_init, env_seed=0, num_train_episode=10000,
            num_val_episode=2000, num_episode_between_val=200, hooks=None,
            verbose=False):
    """Train the agent."""
    params = self.params
    do_early_stopping = params['learning']['patience'] >= 0
    patience = params['learning']['patience']  # for early stopping

    best_reward = float('-inf')
    best_episode = -1

    train_env = env_init()
    train_env.seed(env_seed)

    test_env = env_init()
    test_env.seed(env_seed - 1)

    if verbose:
      log_in_training = logging.info
    else:
      def log_in_training(*args, **kwargs):
        pass

    early_stopping_dir = '{}/{}'.format(EARLY_STOPPING_TEMP_DIR, self.scope)
    makedirs(early_stopping_dir)
    early_stopping_path = early_stopping_dir + '/temp'

    for episode in range(num_train_episode):
      # train
      start = time.time()
      if hooks is not None:
        self.sess.run(hooks)
      ##
      rollouts, _ = simulate_rollouts(self, train_env, 1)
      self.apply_gradients_from_buffer(rollouts[0], episode)
      ## 
      train_time = time.time() - start

      # evaluation
      if num_episode_between_val > 0 and episode % num_episode_between_val == 0:
        start = time.time()
        rollouts, arm_probas = simulate_rollouts(
            self, test_env, num_val_episode)
        metrics = metrics_from_rollouts(rollouts, arm_probas)
        val_time = time.time() - start

        mean_reward = metrics['mean_reward']
        mean_regret = metrics['mean_regret']

        if self.writer is not None:
          global_episode = self.sess.run(self.network.global_step)

          # last_dim = [reward, action, value, proba[0], proba[1]]
          mean_action = np.mean(rollouts[:, :, 1])
          mean_value = np.mean(rollouts[:, :, 2])
          # mean_p_left = np.mean(rollouts[:, :, 3])
          mean_p_right = np.mean(rollouts[:, :, 4])
          val_summary = tf.Summary(value=[
              tf.Summary.Value(tag='train/runtime_per_episode',
                               simple_value=train_time),
              tf.Summary.Value(tag='val/runtime', simple_value=val_time),
              tf.Summary.Value(tag='val/reward', simple_value=mean_reward),
              tf.Summary.Value(tag='val/regret', simple_value=mean_regret),
              tf.Summary.Value(tag='val/action', simple_value=mean_action),
              tf.Summary.Value(tag='val/value', simple_value=mean_value),
              tf.Summary.Value(tag='val/p_right', simple_value=mean_p_right),
              # tf.Summary.Value(tag='val/p_left', simple_value=mean_p_left),
          ])
          self.writer.add_summary(val_summary, global_episode)

        log_in_training(
            'ep=%d, reward=%.3f, regret=%.3f (train %.2f s, val %.2f s)',
            episode, mean_reward, mean_regret, train_time, val_time)

        # early stopping
        if do_early_stopping:
          if metrics['mean_reward'] > best_reward:
            self.save(early_stopping_path)
            patience = params['learning']['patience']  # reset
            best_reward = metrics['mean_reward']
            best_episode = episode
          elif patience > 0:
            patience -= 1
          else:  # end training due to early stopping
            log_in_training('Early stopping at episode %d', episode)
            break

    if do_early_stopping:  # final evaluation
      rollouts, arm_probas = simulate_rollouts(self, test_env, num_val_episode)
      metrics = metrics_from_rollouts(rollouts, arm_probas)

      if metrics['mean_reward'] > best_reward:
        best_episode = episode
        # no need to load best network variable values since already the best
      else:
        self.load(early_stopping_path)

      try:
        shutil.rmtree(EARLY_STOPPING_TEMP_DIR)
      except:
        pass

      log_in_training('Completed training at episode %d, best episode was %d',
                      episode, best_episode)
    else:
      log_in_training('Completed training at episode %d', episode)

    # need these initial previous values for first action step
    self.reset()

  def apply_gradients_from_buffer(self, episode_buffer, episode):
    params = self.params

    prev_rewards, prev_actions, actions, values, returns, advantages = \
        self._process_buffer(episode_buffer, gamma=params['learning']['gamma'])

    hidden_state = self.network.init_state
    ops = self.network.optimize_ops
    actions_in, returns_in, advantages_in = self.network.optimize_inputs
    feed_dict = {
        actions_in: actions,
        returns_in: returns,
        advantages_in: advantages,
        self.network.prev_rewards: prev_rewards,
        self.network.prev_actions: prev_actions,
        self.network.state_inputs[0]: hidden_state[0],
        self.network.state_inputs[1]: hidden_state[1],
    }

    if SAFETY_CHECKS:
      self.sess.run(
          self.network.vp_assert_ops,
          feed_dict={
              self.network.prev_rewards: prev_rewards,
              self.network.prev_actions: prev_actions,
              self.network.state_inputs[0]: hidden_state[0],
              self.network.state_inputs[1]: hidden_state[1],
              self.network.vp_assert_inputs[0]: values,
              self.network.vp_assert_inputs[1]: episode_buffer[:, 3:5],
          })

    _, global_episode, summary = self.sess.run(ops, feed_dict)
    if self.writer is not None:
      self.writer.add_summary(summary, global_episode)

  @staticmethod
  def _process_buffer(episode_buffer, gamma=1., td_error=True, bootstrap=0.):
    
    def _discount(x):
      return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # [reward, action, value, proba[0], proba[1], time]
    rewards = episode_buffer[:, 0]
    actions = episode_buffer[:, 1]
    values = episode_buffer[:, 2]

    prev_rewards = np.concatenate([[0], rewards[:-1]])
    prev_actions = np.concatenate([[0], actions[:-1]])

    if td_error:
      rewards_strapped = np.concatenate((rewards, [bootstrap]))
      returns = _discount(rewards_strapped)[:-1]
      value_strapped = np.concatenate((values, [bootstrap]))
      advantages = rewards + gamma * value_strapped[1:] - value_strapped[:-1]
      advantages = _discount(advantages)
    else:
      returns = _discount(rewards)
      advantages = values - returns

    return prev_rewards, prev_actions, actions, values, returns, advantages


class Architecture(object):
  """TODO"""

  def __init__(self, sess, params=None, scope='architecture'):
    self.params = self.default_params()
    if params:
      recursively_update_dict(self.params, params)

    self.sess = sess
    batch_size = 1

    with tf.variable_scope(scope):
      self.global_step = tf.Variable(0, trainable=False, name='global_step')

      self.prev_actions = tf.placeholder(tf.int32, [None], name='prev_actions')
      self.prev_rewards = tf.placeholder(
          tf.float32, [None], name='prev_rewards')

      prev_rewards = tf.expand_dims(self.prev_rewards, -1)
      prev_actions_hot = tf.one_hot(
          self.prev_actions, self.params['dim_output'], dtype=tf.float32)

      inputs = tf.expand_dims(
          tf.concat([prev_rewards, prev_actions_hot], -1), 0)

      cell = rnn.BasicLSTMCell(self.params['dim_hidden'], state_is_tuple=True)
      c = tf.placeholder(tf.float32, [batch_size, cell.state_size.c], name='c')
      h = tf.placeholder(tf.float32, [batch_size, cell.state_size.h], name='h')
      self.state_inputs = (c, h)
      self.init_state = tf.Session().run(cell.zero_state(batch_size, tf.float32))

      lstm_outputs, self.state_out = tf.nn.dynamic_rnn(
          cell, inputs, initial_state=rnn.LSTMStateTuple(c, h),
          time_major=False)

      self.lstm_outputs = tf.reshape(
          lstm_outputs, [-1, self.params['dim_hidden']])

      # TODO(jisungk): change weights_initializer to Xavier
      self.value = tf.squeeze(tf.contrib.layers.fully_connected(
          self.lstm_outputs, DIM_VALUE, activation_fn=None,
          weights_initializer=None), name='value')
      self.policy = tf.squeeze(tf.contrib.layers.fully_connected(
          self.lstm_outputs, DIM_POLICY, activation_fn=None,
          weights_initializer=None), name='policy')
      self.proba = tf.nn.softmax(self.policy, name='proba')

    self.variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  @staticmethod
  def default_params():
    return {
        'dim_hidden': 48,  # 48
        'dim_output': 2,
    }

  def forward(self, prev_rewards, prev_actions, hidden_state=None):
    if hidden_state is None:
      hidden_state = self.init_state

    ops = [
        self.proba,
        self.value,
        self.lstm_outputs,
        self.state_out,
    ]

    c, h = self.state_inputs
    feed_dict = {
        self.prev_actions: [prev_actions],
        self.prev_rewards: [prev_rewards],
        c: hidden_state[0],
        h: hidden_state[1],
    }

    return self.sess.run(ops, feed_dict=feed_dict)


class TrainableArchitecture(Architecture):

  def __init__(self, sess, optimizer, master=None, params=None,
               scope='trainable_architecture'):
    self.params = self.default_params()
    if params:
      recursively_update_dict(self.params, params)

    architecture_params = self.params['architecture_params']
    super(TrainableArchitecture, self).__init__(
        sess, architecture_params, scope)

    self.params = params
    self.optimizer = optimizer

    self.network_to_update = self if master is None else master
    self._define_training_ops()

  @staticmethod
  def default_params():
    return A2C.default_params()

  def _define_training_ops(self):
    params = self.params

    actions = tf.placeholder(tf.int32, [None], name='actions')
    returns = tf.placeholder(tf.float32, [None], name='returns')
    advantages = tf.placeholder(tf.float32, [None], name='advantages')

    actions_hot = tf.one_hot(actions, 2, dtype=tf.float32)
    log_probas = tf.squeeze(tf.log(self.proba + LOG_OFFSET))
    entropy = -1. * tf.reduce_sum(self.proba * log_probas, axis=-1)

    if SAFETY_CHECKS:
      value_exp = tf.placeholder(tf.float32, [None], name='value_2')
      policy_exp = tf.placeholder(tf.float32, [None, 2], name='policy_2')
      self.vp_assert_inputs = (value_exp, policy_exp)
      self.vp_assert_ops = [
          tf.assert_less_equal(tf.reduce_sum(
              self.value - value_exp), EPISLON),
          tf.assert_less_equal(tf.reduce_mean(
              tf.squeeze(self.policy) - policy_exp), EPISLON)
      ]

    policy_loss = -1 * tf.reduce_mean(
        tf.reduce_sum(log_probas * actions_hot, -1) * advantages,
        name='policy_loss')
    
    value_loss = tf.reduce_mean(
        params['learning']['value_coef'] * 0.5 *
        tf.square(self.value - returns), 
        name='value_loss')
    
    entropy_loss = -1. * tf.reduce_mean(
        params['learning']['entropy_coef'] * entropy, 
        name='entropy_loss')

    loss = tf.identity(policy_loss + value_loss + entropy_loss, 
      name='loss')

    grads = tf.gradients(loss, self.variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 50.)

    increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    # for tensorboard
    train_summary = tf.summary.merge([
        tf.summary.scalar('train/value_loss', value_loss),
        tf.summary.scalar('train/policy_loss', policy_loss),
        tf.summary.scalar('train/entropy_loss', entropy_loss),
        tf.summary.scalar('train/loss', loss),
    ])

    self.optimize_inputs = (actions, returns, advantages)
    self.optimize_ops = [
        self.optimizer.apply_gradients(
            zip(clipped_grads, self.network_to_update.variables)),
        increment_global_step,
        train_summary,
    ]



def simulate_rollouts(agent, env, num_episode=1000):
  # [action, value, reward, time]
  rollouts = -np.ones((num_episode, env._num_step, 5))
  arm_probas = []
  for i in range(num_episode):
    observation = env.reset()
    agent.reset()
    action, reward, done = 0, 0, False
    j = 0
    while not done:
      try:
        step_dict = agent.act(observation, reward, done, verbose=True)
        action = step_dict['action']
        value = step_dict['value']
        proba = step_dict['proba']
        assert j == observation['time']
      except TypeError:
        action = agent.act(observation, reward, done)
        value = -1
        proba = [-1, -1]

      observation, reward, done, _ = env.step(action)
      rollouts[i][j] = [reward, action, value, proba[0], proba[1]]
      j += 1

    arm_probas.append(env._probas)
  return rollouts, arm_probas