import os
import tensorflow as tf
import numpy as np
import scipy

class dependent_bandit():

  def __init__(self):
    self.num_actions = 2
    self.reset()
    self.episode_len = 50

  def reset(self):
    self.timestep = 0
    bandit_pr = np.random.choice([0.9,0.1])
    self.bandit = np.array([bandit_pr, 1-bandit_pr])
    # self.bandit = np.array([0.9,0.1])

  def pullArm(self,action):
    #Get a random number.
    self.timestep += 1
    bandit = self.bandit[action]
    draw = np.random.uniform()
    if draw < bandit: reward = 1
    else: reward = 0
    # terminate
    if self.timestep > self.episode_len: 
        terminal_state = True
    else: terminal_state = False
    return self.timestep,reward,terminal_state



class Network():

  def __init__(self):
    """
    """
    self.num_actions = 2
    self.stsize = 40
    self.batch_size = 1
    self.gamma = 0.9
    self.graph = tf.Graph()
    self.env = dependent_bandit()
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None
    

  def build(self):
    with self.graph.as_default():
      # setup input input_placeholders: [r(t-1),action(t-1),obs(t)]
      concat_inputs = self.input_placeholders()
      # forward propagate inputs
      self.value,self.policy = self.RNN(concat_inputs)
      # setup loss
      self.setup_loss()
      
      self.minimizer = tf.train.RMSPropOptimizer(0.0007).minimize(self.loss)
      ## initialize
      self.sess.run(tf.global_variables_initializer())


  def setup_loss(self):
    """ 
    sums range over episode
    """  
    ## Loss functions
    # policy loss: L = A(s,a) * -logpi(a|s)
    pr_action_t = tf.reduce_sum(self.policy * self.actions_t_onehot) 
    self.policy_loss = - tf.reduce_sum(tf.log(pr_action_t + 1e-7) * self.advantages)
    #
    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - self.value)) 
    self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05
    return None


  def RNN(self,concat_inputs):
    """ Architecture design (LSTM for TD):
    """
    # setup cell
    lstm_cell = tf.keras.layers.LSTMCell(self.stsize)
    init_state = lstm_cell.get_initial_state(
                    tf.get_variable('initial_state',
                      shape=[self.batch_size,self.stsize]))
    # layers
    lstm_layer = tf.keras.layers.RNN(lstm_cell,return_sequences=True,return_state=True)
    value_layer = tf.keras.layers.Dense(1,
                    activation=None,
                    kernel_initializer='glorot_uniform')
    policy_layer = tf.keras.layers.Dense(self.num_actions,
                    activation=tf.nn.softmax,
                    kernel_initializer='glorot_uniform')
    # activations
    self.lstm_outputs,self.final_output,self.lstm_state = lstm_layer(concat_inputs,initial_state=init_state)
    self.value = value_layer(self.lstm_outputs)
    self.policy = policy_layer(self.lstm_outputs)
    return self.value,self.policy


  def input_placeholders(self):
    """
    """
    with self.graph.as_default():
      # self.cellstate_ph = tf.placeholder(name='cellstate',
      #       shape=[self.batch_size],
      #       dtype=tf.float32)
      ## loss placeholders
      self.target_value = tf.placeholder(name='target_value_ph',
            shape=[1,None,1],
            dtype=tf.float32) 
      self.advantages = tf.placeholder(name='advantages_ph',
            shape=[None,1],
            dtype=tf.float32)
      ## input placeholders
      self.rewards_tm1 = tf.placeholder(name='rewards_tm1',
            shape=[None,1],
            dtype=tf.float32)
      self.states_t = tf.placeholder(name='states_t',
            shape=[None,1],
            dtype=tf.float32) 
      # onehot actions
      self.actions_tm1 = tf.placeholder(name='actions_tm1',
            shape=[None],
            dtype=tf.int32)
      self.actions_t = tf.placeholder(name='actions_t',
            shape=[None],
            dtype=tf.int32)
      self.actions_t_onehot = tf.one_hot(
            self.actions_t,self.num_actions,
            name='actions_t_onehot',dtype=tf.float32)
      self.actions_tm1_onehot = tf.one_hot(
            self.actions_tm1,self.num_actions,
            name='actions_tm1_onehot',dtype=tf.float32)
      # concat over units dim
      self.concat_inputs = tf.concat([
              [self.rewards_tm1],
              [self.actions_tm1_onehot],
              [self.states_t]
              ],-1)
    return self.concat_inputs


  def unroll_episode(self):
    """
    unroll agent on environment over an episode
    return data from episode 
    """
    ## initialize episode
    # self.env.reset()
    episode_buffer = []
    terminal_state = False
    reward_t = 0
    action_t = 0
    state_t = 0
    ## unroll episode feeding placeholders in online mode
    self.env.reset()
    while terminal_state == False:
      # print('state',state_t)
      action_dist,value_state_t,lstm_state = self.sess.run(
        [self.policy,self.value,self.lstm_state], 
          feed_dict={
            self.rewards_tm1: [[reward_t]], # batch,time,dim
            self.states_t: [[state_t]],
            self.actions_tm1: [action_t]
            }) 
      # Take an action using probabilities from policy network output.
      action_t = np.random.choice([0,1],p=action_dist.squeeze())
      # observe reward and next_state
      state_t, reward_t, terminal_state = self.env.pullArm(action_t)
      # collect episode information in buffer
      episode_buffer.append([state_t, action_t, reward_t, value_state_t])
    return np.array(episode_buffer)

  def update(self,episode_buffer):
    """
    episode_buffer contains [[state,action,reward,value(state)]]
    using rollout data, compute advantage and discounted returns
    """
    ep_states = episode_buffer[:,0:1] 
    ep_actions = episode_buffer[:,1]
    ep_rewards = episode_buffer[:,2:3]
    ep_values = episode_buffer[:,3:4]
    # shifted reward and action
    ep_rewards_tm1 = np.insert(ep_rewards[:-1,:],0,0,axis=0)
    ep_actions_tm1 = np.insert(ep_actions[:-1],0,0)
    # not sure what this is for
    ep_rewards_ = np.insert(ep_rewards,0,-1,axis=0)
    ep_values_ = np.insert(ep_values,0,-1,axis=0)
    # discount and compute advatnage
    ep_discounted_rewards = discount(ep_rewards_,self.gamma)[:-1]
    # print(ep_rewards.sum(),ep_discounted_rewards)
    ep_advantages = ep_rewards + self.gamma * ep_values_[1:] - ep_values_[:-1]
    ep_discounted_advantages = discount(ep_advantages,self.gamma)
    ## update network using data from episode unroll
    feed_dict = {
        self.states_t: ep_states,
        self.actions_t: ep_actions,
        self.rewards_tm1: ep_rewards_tm1,
        self.actions_tm1: ep_actions_tm1,
        self.target_value: [ep_discounted_rewards],
        self.advantages: ep_discounted_advantages,
        }
    ep_loss,pi,_ = self.sess.run([
                      self.loss,
                      self.policy,
                      self.minimizer
                      ],feed_dict=feed_dict)
    # print('l',ep_loss,'r',ep_discounted_rewards.sum())
    return ep_loss,ep_rewards.mean()



def discount(x, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]