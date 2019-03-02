# Modification from (https://medium.com/p/b15b592a2ddf)

import os
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
import helper

# from helper import *

from random import choice
from time import sleep
from time import time


""" Notes from Wang et al., 2017
- initial state learned
- params:
lr = 0.0007
gamma = 0.9 ()
state_value_estimate_cost = 0.05
entropy_cost = 0.05
unroll_depth = (100,300)
optimizer RMSProp
"""

# ### TASK

class dependent_bandit():

  def __init__(self):
    self.num_actions = 2
    self.reset()

  def reset(self):
    self.timestep = 0
    bandit_pr = np.random.choice([0.9,0.1])
    self.bandit = np.array([bandit_pr, 1-bandit_pr])

  def pullArm(self,action):
    #Get a random number.
    self.timestep += 1
    bandit = self.bandit[action]
    draw = np.random.uniform()
    if draw < bandit:
        #return a positive reward.
        reward = 1
    else:
        #return a negative reward.
        reward = 0
    if self.timestep > 99: 
        terminal_state = True
    else: terminal_state = False
    return reward,self.timestep,terminal_state


# ### Actor-Critic Network


class ACNetwork():

  def __init__(self,num_actions,scope,trainer):
    """
    """
    self.num_actions = num_actions
    self.stsize = 40
    self.batch_size = 1
    self.mygraph = mygraph = tf.Graph()

    with mygraph.as_default(),tf.variable_scope(scope):
      
      # setup input placeholders: [r(t-1),action(t-1),obs(t)]
      concat_input = self.setup_placeholders()
      
      # LSTM for TD 
      lstm_cell = tf.keras.layers.LSTMCell(self.stsize)
      init_state = lstm_cell.get_initial_state(self.cellstate_ph)
      lstm_layer = tf.keras.layers.RNN(lstm_cell,return_sequences=True,return_state=True)
      self.lstm_outputs,self.final_output,self.lstm_state = lstm_layer(concat_input,initial_state=init_state)

      policy_layer = tf.keras.layers.Dense(self.num_actions,
                      activation=tf.nn.softmax,
                      kernel_initializer='glorot_uniform')
      value_layer = tf.keras.layers.Dense(1,
                      activation=None,
                      kernel_initializer='glorot_uniform')

      self.policy = policy_layer(lstm_outputs)
      self.value = value_layer(lstm_outputs)

      # Only the worker network need ops for loss functions and gradient updating.
      """ 
      unsure whats this section is doing. I suspect:
      related to A3C, where different workers unroll on different versions of the environment
        and together update the 'global' shared parameters
      """
      if scope != 'global':
        self.target_v = tf.placeholder(shape=[None],dtype=tf.float32) # discounted_reward
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)  # A(s,a) = Q(s,a) - V(S)
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        # Loss functions
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(
                                  self.target_v - tf.reshape(self.value,[-1])
                                  )) # does MSE vs SSE make a difference here?
        self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.gradients = tf.gradients(self.loss,local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,50.0)
        
        # Apply local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


  def setup_placeholders(self):
    self.cellstate_ph = tf.placeholder(
          shape=[self.batch_size,],
          dtype=tf.float32)
    self.prev_rewards = tf.placeholder(
          shape=[self.batch_size,None,1],
          dtype=tf.float32)
    self.timestep = tf.placeholder(
          shape=[self.batch_size,None,1],
          dtype=tf.float32) 
    self.prev_actions = tf.placeholder(
          shape=[self.batch_size,None],
          dtype=tf.int32)
    self.actions = tf.placeholder(
          shape=[self.batch_size,None],
          dtype=tf.int32)
    # onehot actions
    self.actions_onehot = tf.one_hot(
          self.actions,self.num_actions,
          dtype=tf.float32)
    self.prev_actions_onehot = tf.one_hot(
          self.prev_actions,self.num_actions,
          dtype=tf.float32)
    # concat over units dim
    self.concat_input = concat_input = tf.concat(
      [self.prev_rewards,self.prev_actions_onehot,self.timestep],-1)
    return concat_input


# ### Worker Agent


class Worker():

  def __init__(self,game,name,num_actions,global_episodes):
    """
    Logic of interacting with environment
    """
    self.name = "worker_" + str(name)
    self.number = name        
    self.trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    self.global_episodes = global_episodes
    self.increment = self.global_episodes.assign_add(1)
    self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

    #Create the local copy of the network and the tensorflow op to copy global paramters to local network
    self.local_AC = ACNetwork(num_actions,self.name,trainer)
    self.update_local_ops = update_target_graph('global',self.name)        
    self.env = dependent_bandit('uniform')

  def train(self,episode_buffer,sess,gamma,bootstrap_value):
    """
    """
    episode_buffer = np.array(episode_buffer)
    actions = episode_buffer[:,0]
    rewards = episode_buffer[:,1]
    timesteps = episode_buffer[:,2]
    prev_rewards = [0] + rewards[:-1].tolist()
    prev_actions = [0] + actions[:-1].tolist()
    values = episode_buffer[:,4]

    self.pr = prev_rewards
    self.pa = prev_actions
    # Here we take the rewards and values from the episode_buffer, and use them to 
    # generate the advantage and discounted returns. 
    # The advantage function uses "Generalized Advantage Estimation"
    self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
    discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
    advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
    advantages = discount(advantages,gamma)

    # Update the global network using gradients from loss
    # Generate network statistics to periodically save
    rnn_state = self.local_AC.state_init
    feed_dict = {
        self.local_AC.target_v: discounted_rewards,
        self.local_AC.prev_rewards: np.vstack(prev_rewards),
        self.local_AC.prev_actions: prev_actions,
        self.local_AC.actions: actions,
        self.local_AC.timestep: np.vstack(timesteps),
        self.local_AC.advantages: advantages,
        self.local_AC.state_in[0]: rnn_state[0],
        self.local_AC.state_in[1]: rnn_state[1]
        }
    v_l,p_l,e_l,g_n,v_n,_ = sess.run([
        self.local_AC.value_loss,
        self.local_AC.policy_loss,
        self.local_AC.entropy,
        self.local_AC.grad_norms,
        self.local_AC.var_norms,
        self.local_AC.apply_grads
        ],
        feed_dict=feed_dict)
    return v_l / len(episode_buffer),p_l / len(episode_buffer),e_l / len(episode_buffer), g_n,v_n

  def work(self,gamma,sess,coord,saver,train):
    """
    training loops
    """
    episode_count = sess.run(self.global_episodes)
    total_steps = 0
    print("Starting worker " + str(self.number))
    with sess.as_default(), sess.graph.as_default():  

      while not coord.should_stop(): # is this multiple episodes of training?
        sess.run(self.update_local_ops) # unsure what this does
        # initialize episode
        episode_buffer = []
        terminal_state = False
        reward_t = 0
        action_t = 0
        state_t = 0
        self.env.reset()
        rnn_state = self.local_AC.state_init

        while terminal_state == False:
          
          action_dist,value_state_t,lstm_state = sess.run(
            [self.local_AC.policy,self.local_AC.value,self.local_AC.lstm_state], 
              feed_dict={
                self.local_AC.prev_rewards:[[reward_t]],
                self.local_AC.timestep:[[t]],
                self.local_AC.prev_actions:[action],
                self.local_AC.cellstate_ph: lstm_state
                })
          lstm_state = lstm_state 
          # Take an action using probabilities from policy network output.
          action_t = np.random.choice(action_dist[0],p=action_dist[0])
          action_t = np.argmax(action_dist == action_t)
          # observe reward and next_state
          reward_t, terminal_state, state_t = self.env.pullArm(action_t)
          # collect episode information in buffer
          episode_buffer.append([action_t,reward_t,state_t,terminal_state,value_state_t])

        # Update the network using the experience buffer at the end of the episode.
        if len(episode_buffer) != 0 and train == True:
          v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

        if self.name == 'worker_0':
          sess.run(self.increment)
        episode_count += 1


"""
roll out an entire episode
"""