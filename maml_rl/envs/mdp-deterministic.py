"""
this file is different from mdp-my.
reward function in mdp-my is stochastic. but in this file, i.e., mdp-test, reward function is deterministic.
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


class TabularMDPEnv(gym.Env):

    def __init__(self, num_states, num_actions, task={}):
        super(TabularMDPEnv, self).__init__()
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
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            # x,y=random.randint(0,4),random.randint(0,4)
            # x, y = 2, 2
            """env[x][y]=-1 is the destination"""
            x,y= random.randint(3,4),random.randint(3,4)
            env = self.random_env_matrix()
            
            # x,y = 4,3
            x,y = 4,3
            env[x][y] = -1
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_ in range(self.num_states):
                        x_now, y_now = int(s/wid), s % wid
                        x_next, y_next = int(s_/wid),s_ % wid

                        def stop():
                            if s == s_:
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

                        def left():
                            """
                            Judge:
                            (1) if the current state is on the left_border, it cannot move left;
                            (2) if the left state is blocked, it cannot move left.
                            """
                            if y_now == 0:
                                block_flag = 1
                            elif env[x_now][y_now-1] == 1:
                                block_flag = 1
                            else:
                                block_flag = 0

                            if (x_now == x_next and y_now == (y_next+1) and block_flag != 1)\
                                    or (x_now == x_next and y_now == y_next and block_flag == 1):
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

                        def up():
                            if x_now == 0:
                                block_flag = 1
                            elif env[x_now-1][y_now] == 1:
                                block_flag = 1
                            else:
                                block_flag = 0
                            if (x_now == (x_next+1) and y_now == y_next and block_flag != 1)\
                                    or (x_now == x_next and y_now == y_next and block_flag == 1):
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

                        def right():
                            if y_now == wid-1:
                                block_flag = 1
                            elif env[x_now][y_now+1] == 1:
                                block_flag = 1
                            else:
                                block_flag = 0
                            if (x_now == x_next and y_now == (y_next - 1) and block_flag != 1)\
                                    or (x_now == x_next and y_now == y_next and block_flag == 1):
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

                        def down():
                            if x_now == hei-1:
                                block_flag = 1
                            elif env[x_now + 1][y_now] == 1:
                                block_flag = 1
                            else:
                                block_flag = 0
                            if (x_now == (x_next-1) and y_now == y_next and block_flag != 1)\
                                    or (x_now == x_next and y_now == y_next and block_flag == 1):
                                transitions[i][s][a][s_] = 1.0
                            else:
                                transitions[i][s][a][s_] = 0.0

                        def default():
                            print('fuck you')
                            pass

                        switch={0:stop, 1:left, 2:up, 3:right, 4:down,}
                        switch.get(a,default)()
                        # """
                        # jude the s_ is whether or not blocked.
                        # """
                        # x, y = int(s_ / wid), s_ % wid
                        # block_flag=env[x][y]
                        # # action stop.
                        # if a == 0:
                        #     if s == s_:
                        #         transitions[i][s][a][s_] = 1.0
                        #     else:
                        #         transitions[i][s][a][s_] = 0.0
                        # # action left.
                        # elif a == 1:
                        #     if s == (s_ - 1) and (s % wid) != 4:
                        #         transitions[i][s][a][s_] = 1.0
                        #     elif (s % wid) == 4 and s == s_:
                        #         transitions[i][s][a][s_] = 1.0
                        #     else:
                        #         transitions[i][s][a][s_] = 0.0
                        # # action up.
                        # elif a == 2:
                        #     if s == (s_ - wid) and s < (wid * (wid - 1)) :
                        #         transitions[i][s][a][s_] = 1.0
                        #     elif s == s_ and s >= (wid * (wid - 1)):
                        #         transitions[i][s][a][s_] = 1.0
                        #     else:
                        #         transitions[i][s][a][s_] = 0.0
                        # # action right.
                        # elif a == 3:
                        #     if s == (s_ + 1) and (s % wid) != 0:
                        #         transitions[i][s][a][s_] = 1.0
                        #     elif (s % wid) == 0 and s == s_:
                        #         transitions[i][s][a][s_] = 1.0
                        #     else:
                        #         transitions[i][s][a][s_] = 0.0
                        # # action down.
                        # else:
                        #     if s == (s_ + wid) and s >= wid:
                        #         transitions[i][s][a][s_] = 1.0
                        #     elif s < wid and s == s_:
                        #         transitions[i][s][a][s_] = 1.0
                        #     else:
                        #         transitions[i][s][a][s_] = 0.0

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    x, y = int(s / wid), s % wid
                    if env[x][y] == -1 and a == 0:
                        rewards_mean[i][s][a] = 10
                    else:
                        rewards_mean[i][s][a] = -0.5
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
