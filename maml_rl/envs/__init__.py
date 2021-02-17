from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# TabularMDP-for-self-adaptive-system
# ----------------------------------------

register(
    'TabularMDP-v1',
    entry_point='maml_rl.envs.mdp-my:TabularMDPEnv',
    kwargs={'num_states': 25, 'num_actions': 5},
    max_episode_steps=25
)


# MY experiment!!!!!!!!!!!!!!!!!!!!
# Test the meta algorithm, this is a deterministic reward function mdp.

register(
    'TabularMDPDeterministic-v0',
    entry_point='maml_rl.envs.mdp-deterministic:TabularMDPEnv',
    kwargs={'num_states': 81, 'num_actions': 5},
    max_episode_steps=50
)

# this is very simple grid environment, it is based on MDP
register(
    'SimpleMDP-v0',
    entry_point='maml_rl.envs.mdp-example:SimpleMDP',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=11
)

# this is 10x10 grid environment, it is based on MDP.
register(
    'ComplexMDP-v0',
    entry_point='maml_rl.envs.mdp-complex:ComplexMDP',
    kwargs={'num_states': 82, 'num_actions': 5},
    max_episode_steps=18
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v1',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
