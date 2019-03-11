

policy_loss = -1 * tf.reduce_mean(
  tf.reduce_sum(log_probas * actions_hot, -1) * advantages,
  name='policy_loss')
value_loss = tf.reduce_mean(
  params['learning']['value_coef'] * 0.5 *
  tf.square(self.value - returns), name='value_loss')
entropy_loss = -1. * tf.reduce_mean(
  entropy_coef * entropy, name='entropy_loss')

loss = tf.identity(policy_loss + value_loss + entropy_loss, name='loss')



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