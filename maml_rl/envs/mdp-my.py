"""
this is a maze environment size 5x5

1. warning:

the reward function here is stochastic!!!
in TabularMDPEnv.step:
reward = self.np_random.normal(mean, 1.0)

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

    def random_env_matrix(self):
        env=np.array([
            [0,0,0,0,0],
            [0,0,0,0,0],
            [1,1,1,1,1],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ])
        return env

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
            # x,y=random.randint(0,4),random.randint(0,4)
            x,y=4,4
            env=self.random_env_matrix()
            env[x][y]=-1
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_ in range(self.num_states):
                        if a==0:
                            if s==s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0
                        elif a==1:
                            if s==(s_-1) and (s%wid)!=4:
                                transitions[i][s][a][s_] = 1.0
                            elif (s%wid)==4 and s==s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0
                        elif a==2:
                            if s==(s_-wid) and s<(wid*(wid-1)):
                                transitions[i][s][a][s_] = 1.0
                            elif s==s_ and s>=(wid*(wid-1)):
                                transitions[i][s][a][s_]=1.0
                            else:
                                transitions[i][s][a][s_] = 0.0
                        elif a==3:
                            if s==(s_+1) and (s%wid)!=0:
                                transitions[i][s][a][s_] = 1.0
                            elif (s%wid)==0 and s==s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0
                        else:
                            if s==(s_+wid) and s>=wid:
                                transitions[i][s][a][s_] = 1.0
                            elif s<wid and s==s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    x,y=int(s/wid), s%wid
                    if env[x][y]==-1 and a==0:
                        rewards_mean[i][s][a]= 10
                    else:
                        rewards_mean[i][s][a] = 10
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
        reward = self.np_random.normal(mean, 1.0)

        self._state = self.np_random.choice(self.num_states,
            p=self._transitions[self._state, action])
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation, reward, False, {'task': self._task}
