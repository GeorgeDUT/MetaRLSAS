"""
this file is based on mdp-simple.py
"""
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

import random
import time

MAPWID = 9
MAPHEI = 9
TIME = 0


class ComplexMDP(gym.Env):

    def __init__(self, num_states, num_actions, task={}):
        super(ComplexMDP, self).__init__()
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        # env = np.array([
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        # ])
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
            env = self.random_env_matrix()
            """environment state"""
            # block_x1, block_y1 = random.randint(0,5), random.randint(1,2)
            # block_x2, block_y2 = random.randint(0,5), random.randint(1,2)
            # block_x3, block_y3 = random.randint(0,5), random.randint(1,2)
            # block_x4, block_y4 = random.randint(0,5), random.randint(1,2)
            # block_x1, block_y1 = 0, 3
            # block_x2, block_y2 = 1, 5
            # block_x3, block_y3 = 5, 3
            # block_x4, block_y4 = 6, 3
            # env[block_x1][block_y1] = 1
            # env[block_x2][block_y2] = 1
            # env[block_x3][block_y3] = 1
            # env[block_x4][block_y4] = 1

            """system action"""
            # system_block_x, system_block_y = random.choice([[3,4],[4,3],[3,3]])
            # system_block_x, system_block_y = 3,4
            # env[system_block_x][system_block_y] = 1

            """user goal"""
            """env[x][y]=-1 is the destination"""
            # x,y= random.randint(3,6),random.randint(3,6)
            x, y = 4,4
            env[x][y] = -1

            for jj in range(self.num_states):
                for kk in range(self.num_states):
                    for aa in range(self.num_actions):
                        transitions[i][jj][aa][kk] = 0.0

            for this_state in range(self.num_states-1):
                # stop 0, up 1, down 2, left 3, right 4.
                transitions[i][this_state][0][this_state] = 1.0
                # up
                next_state = (this_state-MAPWID if (this_state-MAPWID>=0 and
                            env[int((this_state-MAPWID)/MAPWID)][(this_state-MAPWID)%MAPWID]!=1)
                              else this_state)
                transitions[i][this_state][1][next_state] = 1.0
                # down
                next_state = (this_state + MAPWID if (this_state + MAPWID < (self.num_states - 1) and
                                                      env[int((this_state + MAPWID) / MAPWID)][
                                                          (this_state + MAPWID) % MAPWID] != 1)
                              else this_state)
                transitions[i][this_state][2][next_state] = 1.0
                # left
                next_state = (this_state-1 if ((this_state%MAPWID !=0) and
                                                      env[int((this_state-1) / MAPWID)][
                                                          (this_state-1) % MAPWID] != 1)
                              else this_state)
                transitions[i][this_state][3][next_state] = 1.0
                # right
                next_state = (this_state + 1 if ((this_state+1)%MAPWID !=0 and
                                                 env[int((this_state+1) / MAPWID)][
                                                     (this_state+1) % MAPWID] != 1)
                              else this_state)
                transitions[i][this_state][4][next_state] = 1.0

            # self.num_states is terminal state
            transitions[i][self.num_states-1][0][self.num_states-1] = 1.0
            transitions[i][self.num_states-1][1][self.num_states-1] = 1.0
            transitions[i][self.num_states-1][2][self.num_states-1] = 1.0
            transitions[i][self.num_states-1][3][self.num_states-1] = 1.0
            transitions[i][self.num_states-1][4][self.num_states-1] = 1.0

            target_state = x*MAPWID+y
            for state in range (self.num_states-1):
                for action in range (self.num_actions):
                    transitions[i][target_state][action][state] = 0.0
            transitions[i][target_state][0][self.num_states-1] = 1.0
            transitions[i][target_state][1][self.num_states-1] = 1.0
            transitions[i][target_state][2][self.num_states-1] = 1.0
            transitions[i][target_state][3][self.num_states-1] = 1.0
            transitions[i][target_state][4][self.num_states-1] = 1.0

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
    env = ComplexMDP(10,5,task)
    a = env.sample_tasks(1)
    # print(a[0].get('rewards_mean'))