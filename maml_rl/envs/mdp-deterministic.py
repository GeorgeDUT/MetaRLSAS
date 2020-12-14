"""
this file is different from mdp-my.
reward function in mdp-my is stochastic. but in this file, i.e., mdp-test, reward function is deterministic.
"""
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

import random


class TabularMDPEnv(gym.Env):

    def __init__(self, num_states, num_actions, task={}):
        super(TabularMDPEnv, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0.0,
            high=1.0, shape=(num_states,), dtype=np.float32)

        self._task = task
        self._transitions = task.get('transitions', np.full((num_states,
            num_actions, num_states), 1.0 / num_states, dtype=np.float32))
        self._rewards_mean = task.get('rewards_mean', np.zeros((num_states,
            num_actions), dtype=np.float32))
        self._state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def sample_tasks(self, num_tasks):
    #     """
    #     generate the transition of the environment T(s,a,s')
    #     return list [[task_1],...,[task_n]]
    #     """
    #     transitions = self.np_random.dirichlet(np.ones(self.num_states),
    #         size=(num_tasks, self.num_states, self.num_actions))
    #     rewards_mean = self.np_random.normal(1.0, 1.0,
    #         size=(num_tasks, self.num_states, self.num_actions))
    #     tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
    #         for (transition, reward_mean) in zip(transitions, rewards_mean)]
    #     return tasks

    def sample_tasks(self, num_tasks):
        """
        generate the transition of the environment T(s,a,s')
        return list [[task_1],...,[task_n]]
        """
        wid = int(pow(self.num_states,0.5))
        transitions = self.np_random.dirichlet(np.ones(self.num_states),
            size=(num_tasks, self.num_states, self.num_actions))
        rewards_mean = self.np_random.normal(1.0, 1.0,
            size=(num_tasks, self.num_states, self.num_actions))
        for i in range(num_tasks):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_ in range(self.num_states):
                        if a==0:
                            if s==s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0
                        elif a==1:
                            if s==s_:
                                transitions[i][s][a][s_] = 0.0
                            else:
                                transitions[i][s][a][s_] = 1.0
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    rewards_mean[i][s][a]=0
        tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
            for (transition, reward_mean) in zip(transitions, rewards_mean)]
        return tasks


    def reset_task(self, task):
        self._task = task
        self._transitions = task['transitions']
        self._rewards_mean = task['rewards_mean']

    def reset(self):
        # From [1]: "an episode always starts on the first state"
        self._state = 0
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        mean = self._rewards_mean[self._state, action]
        reward = mean

        self._state = self.np_random.choice(self.num_states,
            p=self._transitions[self._state, action])
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation, reward, False, {'task': self._task}
