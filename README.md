# Meta_RL_For_SAS
## TooL
Meta_RL_for_SAS:

This project is using Meta Reinforcement learning to enhance the adaptability of self-learning adaptive system (SLAS).


## Algorithm: MAML
Our basic algorithm is based on MAML:

https://github.com/tristandeleu/pytorch-maml-rl

MAML project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in Pytorch. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]


Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)
Implementation of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in Pytorch. This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.

### Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

##### Requirements
 - Python 3.5 or above
 - PyTorch 1.3
 - Gym 0.15

### Usage

##### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with MAML. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on HalfCheetah-Vel:
```
python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8
```

##### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --config maml-halfcheetah-vel/config.json --policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz --meta-batch-size 20 --num-batches 10  --num-workers 8
```

### References

If you want to cite this implementation of MetaRLSAS:
```
@article{mingyue21ameta,
  author    = {Mingyue Zhang and Jialong Li and Haiyan Zhao and Kenji Tei and Shinichi Honiden and Zhi Jin},
  title     = {{A Meta Reinforcement Learning-based Approach for Online Adaptation}},
  journal   = {Hold on},
  year      = {2022},
  url       = {Hold on}
}
```