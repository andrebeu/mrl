import os
import tensorflow as tf
import numpy as np
import scipy
from scipy import signal

""" TODO
sweep gamma, learning rate, optimizer. 
implement uniform independet arms

-- Ji-Sung
LR 7e-4 - 1e-2
gamma 0.2 - 0.8 (0.6)
stsize 24 - 48

extra:
LR decay
gamma annealing 

read:
generalized advantage estimation (schulman et al., 15)
"""

EPLEN = 30

""" 
need a way to change switch modes between train and eval
"""



class SwitchingDependentBandits():
  """ 
  arms have dependent probabilities (0.2,0.8)
  bandit switches within episodes
    switch can either be controlled by a probability
    or by the state number
  """
  def __init__(self,banditpr=.8,switch_param=0,eplen=EPLEN):
    self.final_state = eplen
    self.banditpr = banditpr
    if switch_param <= 1: 
      self.switch_rule = 'prob' # probabilistic switch
      self.switchpr = switch_param
    elif switch_param > 1: 
      self.switch_rule = 'det' # determinstic switch
      self.switch_state = switch_param
    self.bandit = None
    self.reset()

  def reset(self):
    # random arm setup between episode
    self.bandit = np.array([self.banditpr,1-self.banditpr]) 
    np.random.shuffle(self.bandit)
    self.terminate = False
    self.state = 0
    return None

  def eval_reset(self,banditpr):
    # random arm setup between episode
    self.bandit = np.array([banditpr,1-banditpr]) 
    self.terminate = False
    self.state = 0
    return None

  def switch(self):
    self.bandit = np.roll(self.bandit,1)

  def pullArm(self,action):
    reward = np.random.binomial(1,self.bandit[action])
    self.state += 1
    terminate = self.state >= self.final_state
    # probabilistic switch
    if self.switch_rule=='prob':
      if np.random.binomial(1,self.switchpr):
        self.switch()
    elif self.switch_rule=='det':
      if self.state == self.switch_state:
        self.switch()
    return self.state,reward,terminate




class MRLAgent():

  def __init__(self,stsize=48,gamma=.75,optimizer=None,seed=1):
    """
    """
    self.seed = seed
    self.num_actions = 2
    self.stsize = stsize
    self.batch_size = 1
    self.gamma = gamma
    self.env = None # define env in train or eval method
    self.optimizer = optimizer or tf.train.AdamOptimizer(5e-3)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.seed)
      # forward propagate inputs
      self.value_hat,self.policy = self.RNN()
      # setup loss
      self.loss = self.setup_loss()
      self.minimizer = self.optimizer.minimize(self.loss)
      ## initialize
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver()

  def setup_loss(self):
    """ 
    """  
    self.loss_placeholders() # advantages, returns 
    # value
    self.loss_value = (0.5) * tf.reduce_mean(tf.square(self.value_hat - self.returns_ph))
    # policy
    self.pi_a = tf.reduce_sum(self.policy * self.actions_t_onehot,axis=-1)
    self.loss_policy = (-1) * tf.reduce_mean(tf.log(self.pi_a+1e-7) * self.advantages_ph)
    # entropy
    self.entropy = (-1) * tf.reduce_sum(self.policy * tf.log(self.policy+1e-7),axis=-1)
    self.loss_entropy = (-1) * tf.reduce_mean(self.entropy)
    # overall
    self.loss = self.loss_policy + (0.05)*self.loss_value + (0.01)*self.loss_entropy
    return self.loss

  def RNN(self):
    """ Architecture design (LSTM for TD):
    """
    self.input_placeholders() # [r(t-1),action(t-1),obs(t)]
    ## TF RNN
    init_state = tf.nn.rnn_cell.LSTMStateTuple(self.lstm_cstate,self.lstm_hstate)

    cell = tf.nn.rnn_cell.LSTMCell(self.stsize)
    lstm_outputs,lstm_final_states = tf.nn.dynamic_rnn(
      cell=cell,inputs=self.concat_inputs,initial_state=init_state)
    self.lstm_final_cstate,self.lstm_final_hstate = lstm_final_states
    self.lstm_outputs = lstm_outputs
    # readout
    value_layer = tf.keras.layers.Dense(1,
                    activation=None,
                    kernel_initializer='glorot_uniform',
                    name='value_layer')
    policy_layer = tf.keras.layers.Dense(self.num_actions,
                    activation=tf.nn.softmax,
                    kernel_initializer='glorot_uniform',
                    name='policy_layer')
    value = tf.squeeze(value_layer(lstm_outputs),axis=-1)
    policy = policy_layer(lstm_outputs)
    return value,policy

  def loss_placeholders(self):
    self.returns_ph = tf.placeholder(
          name='returns',
          shape=[1,None,], 
          dtype=tf.float32)
    self.advantages_ph = tf.placeholder(
          name='advantages',
          shape=[1,None,], 
          dtype=tf.float32)
    self.actions_t = tf.placeholder(
          name='actions_t',
          shape=[1,None,],
          dtype=tf.int32)
    self.actions_t_onehot = tf.one_hot(
          self.actions_t,self.num_actions,
          name='actions_t_onehot',dtype=tf.float32)
    return None

  def input_placeholders(self):
    ## input placeholders
    self.prev_rewards = tf.placeholder(
          name='prev_rewards',
          shape=[1,None,],
          dtype=tf.float32)
    self.states_t = tf.placeholder(
          name='states_t',
          shape=[1,None,],
          dtype=tf.float32) 
    # onehot actions
    self.prev_actions = tf.placeholder(
          name='prev_actions',
          shape=[1,None,],
          dtype=tf.int32)
    self.prev_actions_onehot = tf.one_hot(
          self.prev_actions,self.num_actions,
          name='prev_actions_onehot',dtype=tf.float32)
    # concat over units dim
    # self.concat_inputs = tf.concat([
    #         tf.expand_dims(self.prev_rewards,-1),
    #         self.prev_actions_onehot,
    #         tf.expand_dims(self.states_t,-1)
    #         ],axis=-1)
    self.concat_inputs = tf.concat([
            tf.expand_dims(self.prev_rewards,-1),
            self.prev_actions_onehot
            ],axis=-1)
    # init cell state 
    self.lstm_cstate = tf.placeholder(
                        name='lstm_cstate_ph',
                        shape=[1,self.stsize],
                        dtype=tf.float32)
    self.lstm_hstate = tf.placeholder(
                        name='lstm_hstate_ph',
                        shape=[1,self.stsize],
                        dtype=tf.float32)
    return None
 
  ## Training and evaluating

  def unroll_episode(self):
    """ online agent-environment interaction
    unroll agent online (i.e. one timestep at a time) over an episode
    return (int) data from episode 
      [state,action,reward,value(state)]_i for i [0,eplen)
    """
    ## initialize episode
    episode_buffer = []
    ep_buffer = -1*np.ones([EPLEN,4])
    terminate = False
    action_t = 0
    reward_t = 0
    state_tp1 = 0
    
    assert self.env.state==0
    ## unroll episode feeding placeholders in online mode
    lstm_cstate,lstm_hstate = np.zeros([2,1,self.stsize])
    while terminate == False:
      # include batch and dim dimensions for feeding
      prev_action = np.expand_dims(np.expand_dims(action_t,0),-1)
      prev_reward = np.expand_dims(np.expand_dims(reward_t,0),-1)
      state_t = np.expand_dims(np.expand_dims(state_tp1,0),-1)
      # feed online
      (action_dist,value_state_t,
      lstm_outputs,lstm_cstate,lstm_hstate) = self.sess.run([
        self.policy,self.value_hat,
        self.lstm_outputs,self.lstm_final_cstate,self.lstm_final_hstate], 
          feed_dict={
            self.lstm_cstate: lstm_cstate,
            self.lstm_hstate: lstm_hstate,
            self.states_t: state_t,
            self.prev_rewards: prev_reward, 
            self.prev_actions: prev_action,
            }) 
      # Take an action using probabilities from policy network output.
      action_t = np.random.choice([0,1],p=action_dist.squeeze())
      # observe reward and next_state
      state_tp1, reward_t, terminate = self.env.pullArm(action_t)
      # collect episode (int) data in buffer
      ep_buffer[state_t,:] = np.array([state_t,action_t,reward_t,value_state_t])
    return ep_buffer

  def process_buffer(self,episode_buffer,td_error=True):
    ## -wrap in .process_buffer()- ## 
    states = episode_buffer[:,0]
    actions = episode_buffer[:,1]
    rewards = episode_buffer[:,2]
    ep_values = episode_buffer[:,3]
    ## compute return 
    if td_error:
      rewards_strapped = np.concatenate((rewards, [0.]))
      value_strapped = np.concatenate((ep_values, [0.]))
      returns = _discount(rewards_strapped,self.gamma)[:-1]
      advantages = rewards + self.gamma * value_strapped[1:] - value_strapped[:-1]
      advantages = _discount(advantages,self.gamma)
    else:
      returns = _discount(rewards,self.gamma)
      advantages = ep_values - returns 
    # shifted actions and rewards
    prev_rewards = np.concatenate([[0],rewards[:-1]])
    prev_actions = np.concatenate([[0],actions[:-1]])
    return states,actions,prev_rewards,prev_actions,returns,advantages

  def update(self,episode_buffer):
    """
    episode_buffer contains [state,action,reward,value(state)]_i for i [0,eplen)
    using rollout data, compute advantage and discounted returns
    """
    (states, actions, prev_rewards, prev_actions,
      ep_returns, ep_advantages) = self.process_buffer(episode_buffer)
    ## update network using data from episode unroll
    init_lstm_cstate,init_lstm_hstate = np.zeros([2,1,self.stsize])
    feed_dict = {
      self.lstm_cstate:init_lstm_cstate,
      self.lstm_hstate:init_lstm_hstate,
        self.prev_rewards:[prev_rewards],
        self.prev_actions:[prev_actions],
        self.states_t:[states],
        self.actions_t:[actions],
      self.returns_ph:[ep_returns],
      self.advantages_ph:[ep_advantages],
        } # NB including batch dimension
    # print(ep_values)
    ## -- ##
    # check norms of gradients
    ## -- ##
    _,loss,vloss,ploss,eloss = self.sess.run([self.minimizer,
        self.loss,self.loss_value,self.loss_policy,self.loss_entropy
        ],feed_dict=feed_dict)
    return np.array([loss,vloss,ploss,eloss])

  def train(self,nepisodes_train,switch_param=0,eps=25):
    """
    """
    self.env = SwitchingDependentBandits(switch_param=switch_param)
    print('train env = switching %f'%switch_param)
    train_loss = np.zeros([nepisodes_train,4])
    for ep in range(nepisodes_train):
      if ep%(nepisodes_train/100)==0:
        print(ep/nepisodes_train)
      self.env.reset()
      episode_buffer = self.unroll_episode()
      ep_loss = self.update(episode_buffer)
      train_loss[ep] = ep_loss
    return train_loss

  def eval(self,nepisodes,banditpr,switch_param=15,eplen=EPLEN):
    """ 
    """
    self.env = SwitchingDependentBandits(switch_param=switch_param)
    print('eval env = switching %f'%switch_param)
    rewards_eval = np.ones([nepisodes,eplen])*787
    for i in range(nepisodes):
      self.env.eval_reset(banditpr)
      ep_buf = self.unroll_episode()
      episode_reward = ep_buf[:,2]
      rewards_eval[i] = episode_reward
    return rewards_eval




def _discount(x,gamma):
  """ computes same thing as above, except faster
  """
  return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
