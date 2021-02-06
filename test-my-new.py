"""
python test-my.py --config maml-halfcheetah-vel/config.json
--policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz
--meta-batch-size 1 --num-batches 1  --num-workers 8

-meta-batch-size xxxx --num-batches xxxx

1.1 mdp task return
result {tasks, train_returns, valid_returns}
task[id]['transitions'][s][a][s']
task[id]['train_returns']
train_retupwrns[task-id][episode-id] episode is defined by fast-batch-size in mdp.yaml or xxx.yaml

change config.json can change the policy gradient step.
"""

import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange
import time

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # env = gym.make(config['env-name'], **config['env-kwargs'])
    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []

    # to see the grad0 ~ multi gradient
    grad0_returns,grad1_returns,grad2_returns,grad3_returns = [],[],[],[]
    grad4_returns, grad5_returns, grad6_returns, grad7_returns = [], [], [], []
    grad8_returns, grad9_returns = [], []
    # to see the grad0 ~ multi gradient

    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)

        # to see the grad0 ~ multi gradient
        grad0_returns.append(get_returns(train_episodes[0]))
        grad1_returns.append(get_returns(train_episodes[2]))
        grad2_returns.append(get_returns(train_episodes[4]))
        grad3_returns.append(get_returns(train_episodes[6]))
        grad4_returns.append(get_returns(train_episodes[8]))
        grad5_returns.append(get_returns(train_episodes[10]))
        grad6_returns.append(get_returns(train_episodes[18]))
        grad7_returns.append(get_returns(train_episodes[24]))
        grad8_returns.append(get_returns(train_episodes[27]))
        grad9_returns.append(get_returns(train_episodes[29]))
        logs['grad0_returns'] = np.concatenate(grad0_returns, axis=0)
        logs['grad1_returns'] = np.concatenate(grad1_returns, axis=0)
        logs['grad2_returns'] = np.concatenate(grad2_returns, axis=0)
        logs['grad3_returns'] = np.concatenate(grad3_returns, axis=0)
        logs['grad4_returns'] = np.concatenate(grad4_returns, axis=0)
        logs['grad5_returns'] = np.concatenate(grad5_returns, axis=0)
        logs['grad6_returns'] = np.concatenate(grad6_returns, axis=0)
        logs['grad7_returns'] = np.concatenate(grad7_returns, axis=0)
        logs['grad8_returns'] = np.concatenate(grad8_returns, axis=0)
        logs['grad9_returns'] = np.concatenate(grad9_returns, axis=0)
        # to see the grad0 ~ multi gradient

        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

    # to see the grad0 ~ multi gradient
    value = [0]*11
    value[0] = logs['grad0_returns'].mean()
    value[1] = logs['grad1_returns'].mean()
    value[2] = logs['grad2_returns'].mean()
    value[3] = logs['grad3_returns'].mean()
    value[4] = logs['grad4_returns'].mean()
    value[5] = logs['grad5_returns'].mean()
    value[6] = logs['grad6_returns'].mean()
    value[7] = logs['grad7_returns'].mean()
    value[8] = logs['grad8_returns'].mean()
    value[9] = logs['grad9_returns'].mean()
    value[10] = logs['valid_returns'].mean()
    print(value)
    # to see the grad0 ~ multi gradient


    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
