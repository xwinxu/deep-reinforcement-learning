# module to test mujoco install

import gym

if __name__ == "__main__":
  env = gym.make('FetchPush-v1')
  env.reset()
  for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
