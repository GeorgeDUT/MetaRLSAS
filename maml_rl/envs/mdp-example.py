"""
this file is different from mdp-my.
reward function in mdp-my is stochastic. but in this file, i.e., mdp-test, reward function is deterministic.
the number of states: wid x wid, and another terminal states.
"""
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

import random
import time

MAPWID = 3
MAPHEI = 3
TIME = 0


class SimpleMDP(gym.Env):

    def __init__(self, num_states, num_actions, task={}):
        super(SimpleMDP, self).__init__()
        self.time_consume = 0
        self.num_states = num_states
        self.num_actions = num_actions

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0, shape=(num_states,), dtype=np.float32)

        self._task = task
        self._transitions = task.get('transitions', np.full((num_states,
                                                             num_actions, num_states), 1.0 / num_states,
                                                            dtype=np.float32))
        self._rewards_mean = task.get('rewards_mean', np.zeros((num_states,
                                                                num_actions), dtype=np.float32))
        self._state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_env_matrix(self):
        # env = np.array([
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        #     [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        #
        # ])

        env = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        return env

    def sample_tasks(self, num_tasks):
        """
        generate the transition of the environment T(s,a,s')
        return list [[task_1],...,[task_n]]
        """

        # time
        time_start = time.time()

        wid = MAPWID
        hei = MAPHEI
        transitions = self.np_random.dirichlet(np.ones(self.num_states),
                                               size=(num_tasks, self.num_states, self.num_actions))
        rewards_mean = self.np_random.normal(1.0, 1.0,
                                             size=(num_tasks, self.num_states, self.num_actions))
        for i in range(num_tasks):
            """env[x][y]=-1 is the destination"""
            # x,y= random.randint(0,2),random.randint(2,2)
            x, y = 0,2
            env = self.random_env_matrix()
            env[x][y] = -1
            for jj in range(self.num_states):
                for kk in range(self.num_states):
                    for aa in range(self.num_actions):
                        transitions[i][jj][aa][kk] = 0.0

            transitions[i][0][0][0] = 1.0
            transitions[i][0][4][1] = 1.0
            transitions[i][0][2][3] = 1.0
            transitions[i][0][1][0] = 1.0
            transitions[i][0][3][0] = 1.0

            transitions[i][1][3][0] = 1.0
            transitions[i][1][0][1] = 1.0
            transitions[i][1][4][2] = 1.0
            transitions[i][1][2][4] = 1.0
            transitions[i][1][1][1] = 1.0

            transitions[i][2][3][1] = 1.0
            transitions[i][2][0][2] = 1.0
            transitions[i][2][2][5] = 1.0
            transitions[i][2][1][2] = 1.0
            transitions[i][2][4][2] = 1.0

            transitions[i][3][1][0] = 1.0
            transitions[i][3][0][3] = 1.0
            transitions[i][3][4][4] = 1.0
            transitions[i][3][2][6] = 1.0
            transitions[i][3][3][3] = 1.0

            transitions[i][4][1][1] = 1.0
            transitions[i][4][3][3] = 1.0
            transitions[i][4][0][4] = 1.0
            transitions[i][4][4][5] = 1.0
            transitions[i][4][2][7] = 1.0

            transitions[i][5][1][2] = 1.0
            transitions[i][5][3][4] = 1.0
            transitions[i][5][0][5] = 1.0
            transitions[i][5][2][8] = 1.0
            transitions[i][5][4][5] = 1.0

            transitions[i][6][1][3] = 1.0
            transitions[i][6][0][6] = 1.0
            transitions[i][6][4][7] = 1.0
            transitions[i][6][2][6] = 1.0
            transitions[i][6][3][6] = 1.0

            transitions[i][7][1][4] = 1.0
            transitions[i][7][3][6] = 1.0
            transitions[i][7][0][7] = 1.0
            transitions[i][7][4][8] = 1.0
            transitions[i][7][2][7] = 1.0

            transitions[i][8][2][5] = 1.0
            transitions[i][8][3][7] = 1.0
            transitions[i][8][0][8] = 1.0
            transitions[i][8][1][8] = 1.0
            transitions[i][8][4][8] = 1.0

            # 9 is terminal state
            transitions[i][9][0][9] = 1.0
            transitions[i][9][1][9] = 1.0
            transitions[i][9][2][9] = 1.0
            transitions[i][9][3][9] = 1.0
            transitions[i][9][4][9] = 1.0
            target_state = x*MAPWID+y
            for state in range (self.num_states):
                for action in range (self.num_actions):
                    transitions[i][target_state][action][state] = 0.0
            transitions[i][target_state][0][9] = 1.0
            transitions[i][target_state][1][9] = 1.0
            transitions[i][target_state][2][9] = 1.0
            transitions[i][target_state][3][9] = 1.0
            transitions[i][target_state][4][9] = 1.0

            for s in range(self.num_states-1):
                for a in range(self.num_actions):
                    x, y = int((s) / wid), (s) % wid
                    if env[x][y] == -1 and a == 0:
                        rewards_mean[i][s][a] = 10
                    else:
                        rewards_mean[i][s][a] = -0.5
            for a in range(self.num_actions):
                rewards_mean[i][self.num_states-1][a] = 0
        tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
                 for (transition, reward_mean) in zip(transitions, rewards_mean)]

        # time
        time_end = time.time()
        self.time_consume = self.time_consume + (time_end-time_start)
        print("construct MDP time consume (seconds)", self.time_consume)

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


if __name__ == '__main__':
    task = {}
    env = SimpleMDP(10,5,task)
    a = env.sample_tasks(1)
    print(a[0].get('rewards_mean'))