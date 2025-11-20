"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from this import s
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim  # 动作空间的维度（action dimension），即输出动作的维度
        self.ob_dim = ob_dim  # 观察空间的维度（observation dimension），即输入状态的维度
        self.n_layers = n_layers  # 神经网络隐藏层的数量
        self.size = size  # 每个隐藏层的神经元数量（隐藏层大小）
        self.learning_rate = learning_rate  # 学习率，用于优化器更新参数
        self.training = training  # 是否处于训练模式（布尔值）
        self.nn_baseline = nn_baseline  # 是否使用神经网络作为baseline（用于方差减少）

        # 构建输出动作均值的神经网络
        # 输入：观察（状态），输出：动作的均值
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)  # 将网络移动到指定设备（CPU或GPU）
        
        # logstd: 动作标准差的自然对数（log standard deviation）
        # 作为可学习的参数，用于定义动作分布的标准差
        # 初始化为全零向量，维度等于动作维度
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)  # 将参数移动到指定设备
        
        # 创建Adam优化器，同时优化logstd参数和mean_net的所有参数
        # itertools.chain用于将logstd和mean_net的参数合并到一个参数列表中
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!

        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation.astype(np.float32))
        # 动作均值
        mean_action = self.mean_net(observation)
        # 动作标准差
        std_action = torch.exp(self.logstd)
        # 创建正态分布
        dist = distributions.Normal(mean_action, std_action)

        return dist

    def get_action(self, obs: np.ndarray) -> np.ndarray:

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
    
        # 将 numpy 数组转换为 Tensor
        observation = ptu.from_numpy(observation.astype(np.float32))

        # 动作均值
        mean_action = self.mean_net(observation)
        # 动作标准差
        std_action = torch.exp(self.logstd)
        # 创建正态分布
        dist = distributions.Normal(mean_action, std_action)
        action = dist.sample()

        return ptu.to_numpy(action)


    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        # 转为tensor
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations.astype(np.float32))
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions.astype(np.float32))
        # 输出
        dist = self.forward(observations)
        # 计算负对数损失
        log_probs = dist.log_prob(actions)  # 形状: (batch_size, action_dim)
        log_probs = log_probs.sum(dim=-1)   # 对动作维度求和，形状: (batch_size,)
        loss = -log_probs.mean()            # 对 batch 求平均，得到标量
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
