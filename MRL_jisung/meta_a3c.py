"""Meta Asynchronous Actor-Critic Agents (A3C) Agent.

Author: Ji-Sung Kim (Daw Lab)
Email:  hello (at) jisungkim.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import threading

from absl import logging

import numpy as np
import tensorflow as tf

from .base import Base
from .meta_a2c import A2C
from .meta_a2c import Architecture
from .meta_a2c import TrainableArchitecture
from .utils import copy_vars
from .utils import makedirs
from .utils import metrics_from_rollouts
from .utils import recursively_update_dict
from .utils import simulate_rollouts

EARLY_STOPPING_TEMP_DIR = 'tmp_es'


class A3C(A2C):

  def __init__(self, action_space, sess=None, writer_dir=None, num_worker=6,
               scope=None, params=None):
    self.params = self.default_params()
    if params:
      recursively_update_dict(self.params, params)

    self.sess = sess if sess is not None else tf.Session()
    self.scope = scope if scope is not None else 'a3c'

    architecture_params = self.params['architecture_params']
    master = Architecture(self.sess, architecture_params, scope=self.scope)
    self.network = master

    optimizer_params = self.params['optimizer_params']
    if self.params['optimizer'].lower() == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(**optimizer_params)
    elif self.params['optimizer'].lower() == 'adam':
      optimizer = tf.train.AdamOptimizer(**optimizer_params)

    worker_params = copy.deepcopy(self.params)
    # workers should not use early stopping themselves
    worker_params['learning']['patience'] = -1

    self.workers = []
    for i in range(num_worker):
      worker_scope = 'worker_{}'.format(i)
      curr_writer_dir = writer_dir if i == 0 else None  # only worker_0 writes
      worker = Worker(
          action_space, master, self.sess, optimizer,
          writer_dir=curr_writer_dir, scope=worker_scope, params=worker_params)
      self.workers.append(worker)

    # TODO(jisungk): figure out step
    self.sess.run(tf.global_variables_initializer())

    self.saver = tf.train.Saver(var_list=self.network.variables)
    self.writer = self.workers[0].writer

    self.reset()

  def train(self, env_init, env_seed=0, num_train_episode=10000,
            num_val_episode=2000, num_episode_between_val=200, hooks=None,
            verbose=False):
    """Train the agent."""
    params = self.params
    do_early_stopping = params['learning']['patience'] >= 0
    patience = params['learning']['patience']  # for early stopping
    best_reward = float('-inf')
    best_episode = -1

    test_env = env_init()
    test_env.seed(env_seed - 1)

    early_stopping_dir = '{}/{}'.format(EARLY_STOPPING_TEMP_DIR, self.scope)
    makedirs(early_stopping_dir)
    early_stopping_path = early_stopping_dir + '/temp'

    episode = 0
    while episode < num_train_episode:
      start = time.time()

      # training for this block
      episode_block = min(num_train_episode - episode, num_episode_between_val)
      episode += episode_block

      coord = tf.train.Coordinator()
      worker_threads = []
      for idx_worker, worker in enumerate(self.workers):
        worker_work = lambda: worker.work(
            env_init, env_seed=env_seed + idx_worker, num_episode=episode_block)

        thread = threading.Thread(target=(worker_work))
        thread.start()
        worker_threads.append(thread)
      coord.join(worker_threads)
      train_time = float(time.time() - start) / episode_block

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
          # shape is ((num_episode, env._num_step, 6))
          # last_dim = [reward, action, value, proba[0], proba[1], time]
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
          self.writer.add_summary(val_summary, episode)

        if verbose:
          logging.info(
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
            logging.info('Early stopping at episode %d', episode)
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

      logging.info('Completed training at episode %d, best episode was %d',
                   episode, best_episode)
    else:
      logging.info('Completed training at episode %d', episode)

    # need these initial previous values for first action step
    self.reset()

  def _define_training_ops(self):
    logging.warn('`_define_training_ops` is undefined for A3C master.')

  def _apply_gradients_from_buffer(self, episode_buffer, episode):
    logging.warn('`_apply_gradients_from_buffer` is undefined for A3C master.')

  def _process_buffer(self, episode_buffer, td_error=True, bootstrap=0.):
    logging.warn('`_process_buffer` is undefined for A3C master.')


class Worker(A2C):

  def __init__(self, action_space, master, sess, optimizer, writer_dir=None,
               scope='worker', params=None):
    super(A2C, self).__init__(action_space)

    self.params = self.default_params()
    if params:
      recursively_update_dict(self.params, params)

    self.sess = sess
    self.scope = scope
    self.optimizer = optimizer

    self.network = TrainableArchitecture(
        self.sess, optimizer, master, params=self.params, scope=self.scope)

    if writer_dir is not None:
      self.writer = tf.summary.FileWriter(
          writer_dir, self.sess.graph, flush_secs=10)
    else:
      self.writer = None

    self.update_local_op = copy_vars(master.variables, self.network.variables)

    self.reset()

  def work(self, env_init, env_seed, num_episode):
    update_local_hooks = [self.update_local_op]
    self.train(env_init, env_seed=env_seed, num_train_episode=num_episode,
               num_val_episode=-1,  num_episode_between_val=-1,
               hooks=update_local_hooks, verbose=False)

  def save(self, path):
    logging.warn('`save()` is undefined for A3C Worker.')

  def load(self, path):
    logging.warn('`load()` is undefined for A3C Worker.')
