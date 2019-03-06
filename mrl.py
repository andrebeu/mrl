import os
import tensorflow as tf
import numpy as np
import scipy


""" TODO
- random length episodes during training, fixed length episodes during eval
- loss parameters
- different rand seeds

-- setup param sweep, which takes snapshots (i.e. eval on held out trials) 
  at multiple times during training

- concern: currently seeing very little learning. this might be lack of optimization
  or my implementation might just be learning which arm is more likely under 
  the current random seed
"""



class SwitchingBandits():

  def __init__(self,shiftpr=0.1,armpr=0.9,eplen=75):
    self.banditpr = np.array([armpr,1-armpr])
    self.shiftpr = shiftpr
    self.final_state = eplen
    self.reset()

  def reset(self):
    # random arm setup between episode
    np.random.shuffle(self.banditpr) 
    self.terminate = False
    self.state = 0
    return None

  def pullArm(self,action):
    """ 
    """
    if np.random.binomial(1,self.shiftpr):
      # within episode shift
      self.banditpr = np.roll(self.banditpr,1)
    reward = np.random.binomial(1,self.banditpr[action])
    self.state += 1
    terminate = self.state == self.final_state
    return self.state,reward,terminate


class MRLAgent():

  def __init__(self,stsize=50,gamma=0.9,task=SwitchingBandits(shiftpr=0)):
    """
    """
    self.num_actions = 2
    self.stsize = stsize
    self.batch_size = 1
    self.gamma = gamma
    self.graph = tf.Graph()
    self.env = task
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      self.misc_placeholders()
      # forward propagate inputs
      self.concat_inputs = self.input_placeholders() # [r(t-1),action(t-1),obs(t)]
      self.value,self.policy = self.RNN(self.concat_inputs)
      # setup loss
      self.target_value,self.advantages = self.loss_placeholders()
      self.loss = self.setup_loss()
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
    policy_loss = - tf.reduce_sum(tf.log(pr_action_t + 1e-7) * self.advantages)
    #
    value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - self.value)) 
    entropy_loss = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
    loss = 0.5 * value_loss + policy_loss - entropy_loss * 0.05
    return loss

  def setup_loss_(self):
    """ 
    sums range over episode
    """  
    ## Loss functions
    # Return 
    # policy loss: L = A(s,a) * -logpi(a|s)
    
    return loss

  def RNN(self,concat_inputs):
    """ Architecture design (LSTM for TD):
    """
    # setup cell
    lstm_cell = tf.keras.layers.LSTMCell(self.stsize)
    init_state = lstm_cell.get_initial_state(
                    tf.get_variable('initial_state',trainable=True,
                      shape=[self.batch_size,self.stsize]))
    # network layers
    lstm_layer = tf.keras.layers.RNN(lstm_cell,
                    return_sequences=True,
                    return_state=True)
    value_layer = tf.keras.layers.Dense(1,
                    activation=None,
                    kernel_initializer='glorot_uniform')
    policy_layer = tf.keras.layers.Dense(self.num_actions,
                    activation=tf.nn.softmax,
                    kernel_initializer='glorot_uniform')
    # dropout layers
    value_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    policy_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    # activations
    lstm_outputs,final_output,lstm_state = lstm_layer(concat_inputs,initial_state=init_state)
    value = value_dropout(value_layer(lstm_outputs))
    policy = policy_dropout(policy_layer(lstm_outputs))
    return value,policy

  def loss_placeholders(self):
    with self.graph.as_default():
      ## loss placeholders
      target_value = tf.placeholder(name='target_value_ph',
            shape=[1,None,1],
            dtype=tf.float32) 
      advantages = tf.placeholder(name='advantages_ph',
            shape=[None,1],
            dtype=tf.float32)
      return target_value,advantages

  def input_placeholders(self):
    """
    """
    with self.graph.as_default():
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
      concat_inputs = tf.concat([
              [self.rewards_tm1],
              [self.actions_tm1_onehot],
              [self.states_t]
              ],-1)
    return concat_inputs

  def misc_placeholders(self):
    self.dropout_rate = tf.placeholder(name='dropout_rate',
            shape=[],
            dtype=tf.float32)
    return None
  ## Training and evaluating

  def unroll_episode(self,training=False):
    """
    unroll agent on environment over an episode
    return data from episode 
    """
    ## initialize episode
    episode_buffer = []
    terminal_state = False
    reward_t = 0
    action_t = 0
    state_t = 0
    if training:
      dropout_rate = 0.1
    else:
      dropout_rate = 0.0
    ## unroll episode feeding placeholders in online mode
    self.env.reset()
    while terminal_state == False:
      # print('state',state_t)
      action_dist,value_state_t = self.sess.run(
        [self.policy,self.value], 
          feed_dict={
            self.rewards_tm1: [[reward_t]], # batch,time,dim
            self.states_t: [[state_t]],
            self.actions_tm1: [action_t],
            self.dropout_rate: dropout_rate
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
        self.dropout_rate: 0.0
        }
    ep_loss,pi,_ = self.sess.run([
                      self.loss,
                      self.policy,
                      self.minimizer
                      ],feed_dict=feed_dict)
    # print('l',ep_loss,'r',ep_discounted_rewards.sum())
    return ep_loss,ep_rewards.mean()

  def train(self,nepisodes_train):
    """
    """
    rewards_train = np.zeros([nepisodes_train])
    for ep in range(nepisodes_train):
      if ep%(nepisodes_train/100)==0:
        print(ep/nepisodes_train)
      episode_buffer = self.unroll_episode(training=True)
      _,ep_reward = self.update(episode_buffer)
      rewards_train[ep] = ep_reward
    return rewards_train

  def eval(self,nepisodes_eval):
    """ 
    """
    rewards_eval = np.ones([nepisodes_eval,75])*100
    for i in range(nepisodes_eval):
      ep_buf = self.unroll_episode(training=False)
      r = ep_buf[:,2]
      rewards_eval[i] = r
    return rewards_eval
      

def discount(x, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]