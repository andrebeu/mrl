import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical



class GeneralBandits():
  """ 
  arms have dependent probabilities (0.2,0.8)
  bandit switches within episodes
    switch can either be controlled by a probability
    or by the state number
  """
  def __init__(self,narms=2,banditpr=.8,switch_param=0,eplen=30):
    self.narms=narms
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
    obs = [self.state,action,reward]
    return obs,reward,terminate


class MRLBandit():
  """ metaRL bandit
  dependent arm probabilities e.g. (0.2,0.8)
  """
  def __init__(self,banditpr,eplen,narms=2):
    self.narms=narms
    self.final_state = eplen
    self.banditpr = banditpr
    self.state = None
    self.reset()
    self.shuffle_bestarm = True

  def reset(self):
    # random arm setup between episode
    self.state = 0
    self.terminate = False
    self.banditprs = np.array([self.banditpr,1-self.banditpr]) 
    if self.shuffle_bestarm:
      np.random.shuffle(self.banditprs)
    return None

  def step(self,action):
    self.state += 1
    reward = np.random.binomial(1,self.banditprs[action])
    terminate = self.state > self.final_state
    # form mrl obs
    s_tm1 = self.state-1
    a_t = np.eye(2)[action]
    r_t = reward
    obs_t = np.concatenate([[s_tm1],a_t,[r_t]])
    return obs_t,reward,terminate


Experience = namedtuple('Experience',[
    'tstep','state','action','reward','state_tp1','rnn_state'
])

class ActorCritic(tr.nn.Module):
  
  def __init__(self,indim=4,nactions=2,stsize=18,gamma=0.80,learnrate=0.005,TDupdate=False):
    super().__init__()
    self.indim = indim
    self.stsize = stsize
    self.nactions = nactions
    self.learnrate = learnrate
    self.gamma = gamma
    self.TDupdate = TDupdate
    self.build()
    return None

  def build(self):
    """ concat input [s_tm1,a_tm1,r_tm1]
    """
    # policy parameters
    self.rnn = tr.nn.LSTMCell(self.indim,self.stsize,bias=True)
    self.rnn_st0 = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    self.rnn2val = tr.nn.Linear(self.stsize,1,bias=True)
    self.rnn2pi = tr.nn.Linear(self.stsize,self.nactions,bias=True)
    # optimization
    self.optiop = tr.optim.RMSprop(self.parameters(), 
      lr=self.learnrate
    )
    return None

  def unroll_ep(self,task):
    """ actor logic 
    """
    finalst = False
    task.reset()
    self.h_t,self.c_t = self.rnn_st0
    EpBuff = []
    action = np.random.binomial(1,0.5)
    while not finalst:
      obs,r_t,finalst = task.step(action)
      h_t = self.rnn_step(obs)
      vh_t = self.rnn2val(h_t)
      pih_t = self.rnn2pi(h_t) 
      action = self.act(pih_t)
      exp = Experience('tstep',obs[0],action,r_t,obs[0]+1,h_t)
      EpBuff.append(exp)
    return EpBuff

  def rnn_step(self,obs):
    obs = tr.Tensor(obs).unsqueeze(0)
    self.h_t,self.c_t = self.rnn(obs,(self.h_t,self.c_t))
    return self.h_t

  def act(self,pi_out):
    """ pi_out [batch,nactions] is output of policy head
    """
    pism = pi_out.softmax(-1)
    pidistr = Categorical(pism)
    actions = pidistr.sample()
    return actions

  def eval(self,expD):
    """ """
    data = {}
    ## entropy
    vhat,pact = self.forward(expD['state'])
    pra = pact.softmax(-1)
    entropy = -1 * tr.sum(pra*pra.log2(),-1).mean()
    data['entropy'] = entropy.detach().numpy()
    ## value
    returns = compute_returns(expD['reward']) 
    data['delta'] = np.mean(returns - vhat.detach().numpy())
    return data

  def update(self,expD):
    """ REINFORCE and A2C updates
    given expD trajectory:
     expD = {'reward':[tsteps],'state':[tsteps],...}
    """
    # unpack experience
    rnn_states = tr.cat([*expD['rnn_state']])
    vhat = self.rnn2val(rnn_states)
    pact = self.rnn2pi(rnn_states)
    actions = tr.Tensor(expD['action'])
    reward = expD['reward']
    returns = compute_returns(expD['reward'],gamma=self.gamma) 
    # form RL target
    if self.TDupdate: # actor-critic loss
      delta = tr.Tensor(expD['reward'][:-1])+self.gamma*vhat[1:].squeeze()-vhat[:-1].squeeze()
      delta = tr.cat([delta,tr.Tensor([0])])
    else: # REINFORCE
      delta = tr.Tensor(returns) - vhat.squeeze()
    # form RL loss
    pi = pact.softmax(-1)
    distr = Categorical(pi)
    los_pi = tr.mean(delta*distr.log_prob(actions))
    ent_pi = tr.mean(tr.sum(pi*tr.log(pi),1))
    los_val = tr.square(tr.Tensor(returns) - vhat.squeeze()).mean()
    los = 0.1*los_val-los_pi+0.1*ent_pi
    # update step
    self.optiop.zero_grad()
    los.backward()
    self.optiop.step()
    return None 

# entropy
# self.entropy = (-1) * tf.reduce_sum(self.policy * tf.log(self.policy+1e-7),axis=-1)
# self.loss_entropy = (-1) * tf.reduce_mean(self.entropy)


def compute_returns(rewards,gamma=1.0):
  """ 
  given rewards, compute discounted return
  G_t = sum_k [g^k * r(t+k)]; k=0...T-t
  """ 
  T = len(rewards) 
  returns = np.array([
      np.sum(np.array(
          rewards[t:])*np.array(
          [gamma**i for i in range(T-t)]
      )) for t in range(T)
  ])
  return returns


def unpack_expL(expLoD):
  """ 
  given list of experience (namedtups)
      expLoD [{t,s,a,r,sp}_t]
  return dict of np.arrs 
      exp {s:[],a:[],r:[],sp:[]}
  """
  expDoL = Experience(*zip(*expLoD))._asdict()
  return {k:np.array(v) for k,v in expDoL.items()}

