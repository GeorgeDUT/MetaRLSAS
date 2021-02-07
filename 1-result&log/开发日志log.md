2021/1/29
训练一轮
主要配置 
register(
    'TabularMDPDeterministic-v0',
    entry_point='maml_rl.envs.mdp-deterministic:TabularMDPEnv',
    kwargs={'num_states': 81, 'num_actions': 5},
    max_episode_steps=50
)

# Number of outer-loop updates (ie. number of batches of tasks).
num-batches: 501

# Number of trajectories to sample for each task.
fast-batch-size: 27

训练时间 2小时左右。
all offline training time cost (second) 8386.6758081913
construct MDP time consume (seconds) 580.7706327438354

第二轮训练
construct MDP time consume (seconds) 587.937745809555
501/501 [2:17:55<00:00, 16.52s/it]
all offline training time cost (second) 8275.299672365189



参数设置
max_episode_steps=50 是每个episode的步长
# Number of trajectories to sample for each task.
fast-batch-size: 27
一个任务采27个episode/trajectories


meta-batch-size 5 --num-batches 10


输出数据格式：
avg_train_r 平均训练奖励；
[
[t1,t2,...tn] # 同一个任务采n个episode/trajectories
有 meta-batch-size x --num-batches  个episode
]

avg_valid_r 平均评估奖励；
[
[t1,t2,...tn] # 同一个任务采n个episode/trajectories
有 meta-batch-size x --num-batches  个episode
]



2021/1/30

修改了test-my-plus.py 文件，现在怀疑老的test-my.py无法实现策略在线更新。
参考train.py增加了两条TODO，但似乎还没有效果，下一步计划将policy权值打印出来参考，是否更新了策略。

2020/2/5
修改了test-my-plus.py 文件，把policy权值打印出来看，没有变化：
打印语句如下：
        for name,param in policy.layer1.named_parameters():
            print(name,param)
在train.py中，权值是在变化的。
why？

目前的改动：
1）新建：for-test-my.py ，这个来源于train.py，调用方法train.py 一样，代码中会预先加载已训练的policy
2）先用train.py在mdp-dterministic.py的环境中训练（这时的mdp是随机生成的。 x,y= random.randint(3,4),random.randint(3,4)）。得到policy后用for-test-my.py进行测试，修改mdp-dterministic.py为固定。
新的发现
test-my.py中的 train_episodes[0]，train_episodes[1]，train_episodes[2]，似乎分别表示更新一次梯度，两次梯度，三次梯度得到的结果。

2021/2/5
新训练了 2d-navigation
500/500 [3:08:16<00:00, 22.59s/it]
all offline training time cost (second) 11296.75797700882
在训练好的policy上进行测试：
test-my-new.py
（--meta-batch-size 3 --num-batches 5）30 num-steps gradients:
结果：每一项表示进行1步，2步，4，8，。。。gradients以后的平均奖励。
[-39.929504, -24.976614, -18.99887, -18.210821, -13.7782, -16.784372, -16.842138, -15.849552, -18.467426, -17.847656]

（--meta-batch-size 3 --num-batches 10）
[-40.29515, -39.562428, -20.649496, -15.440778, -15.30632, -18.51892, -19.379017, -18.016151, -20.298206, -19.244917]


2021/2/7
新开发的mdp-exmaple.py，
改文件SimpleMDP提供了一个3x3的小网格，目标点可以随机化生成，没有障碍物。
在3x3的小地图上，元训练100轮后，进行测试
[11.464348, 80.03391, 81.44, 81.42174, 81.44, 81.44, 81.44, 81.44, 81.44, 81.44, 81.44]
81.44
有明显的提升。

小环境下的场景基本调通，可以进行下一步实验




