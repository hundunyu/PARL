#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gym
import paddle.fluid as fluid
import numpy as np
import os
import parl
from atari_agent import AtariAgent
from atari_model import AtariModel
from datetime import datetime
from replay_memory import ReplayMemory, Experience
from parl.utils import tensorboard, logger
from tqdm import tqdm
from utils import get_player

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 3e-4


def run_train_episode(env, agent, rpm):
    total_reward = 0
    all_cost = []
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        context = rpm.recent_state()
        context.append(state)  # 一个完整的context
        context = np.stack(context, axis=0)  # 转list为array
        action = agent.sample(context)
        next_state, reward, isOver, _ = env.step(action)
        rpm.append(Experience(state, action, reward, isOver))
        # start training
        # 大于 MEMORY_WARMUP_SIZE，开始训练
        if rpm.size() > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = rpm.sample_batch(
                    args.batch_size)
                batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
                batch_next_state = batch_all_state[:, 1:, :, :]
                cost = agent.learn(batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_isOver)
                all_cost.append(float(cost))
        total_reward += reward
        state = next_state
        if isOver:
            break
    if all_cost:
        logger.info('[Train]total_reward: {}, mean_cost: {}'.format(
            total_reward, np.mean(all_cost)))
    return total_reward, steps, np.mean(all_cost)


def run_evaluate_episode(env, agent):
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.predict(state)
        state, reward, isOver, info = env.step(action)
        total_reward += reward
        if isOver:
            break
    return total_reward


def main():
    env = get_player(
        args.rom, image_size=IMAGE_SIZE, train=True, frame_skip=FRAME_SKIP)
    test_env = get_player(
        args.rom,
        image_size=IMAGE_SIZE,
        frame_skip=FRAME_SKIP,
        context_len=CONTEXT_LEN)
    rpm = ReplayMemory(MEMORY_SIZE, IMAGE_SIZE, CONTEXT_LEN)
    act_dim = env.action_space.n

    # model: 模型结构定义
    model = AtariModel(act_dim, args.algo)
    # algorithm: 前向网络及反馈学习设计
    if args.algo == 'Double':
        algorithm = parl.algorithms.DDQN(model, act_dim=act_dim, gamma=GAMMA)
    elif args.algo in ['DQN', 'Dueling']:
        algorithm = parl.algorithms.DQN(model, act_dim=act_dim, gamma=GAMMA)
    # agent: 数据交互
    agent = AtariAgent(
        algorithm,
        act_dim=act_dim,
        start_lr=LEARNING_RATE,
        total_step=args.train_total_steps,
        update_freq=UPDATE_FREQ)

    with tqdm(
            total=MEMORY_WARMUP_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        # 预热环节，先准备 MEMORY_WARMUP_SIZE 个 Experience
        # 提供初始数据，为学习和随机采样做充足准备
        while rpm.size() < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_train_episode(env, agent, rpm)
            pbar.update(steps)

    # train
    # 及学习策略
    test_flag = 0
    pbar = tqdm(total=args.train_total_steps)
    total_steps = 0
    max_reward = None
    while total_steps < args.train_total_steps:
        # start epoch
        total_reward, steps, loss = run_train_episode(env, agent, rpm)
        total_steps += steps
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        tensorboard.add_scalar('dqn/score', total_reward, total_steps)
        tensorboard.add_scalar('dqn/loss', loss,
                               total_steps)  # mean of total loss
        tensorboard.add_scalar('dqn/exploration', agent.exploration,
                               total_steps)
        pbar.update(steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            pbar.write("testing")
            eval_rewards = []
            for _ in tqdm(range(3), desc='eval agent'):
                eval_reward = run_evaluate_episode(test_env, agent)
                eval_rewards.append(eval_reward)
            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    total_steps, np.mean(eval_rewards)))
            eval_test = np.mean(eval_rewards)
            tensorboard.add_scalar('dqn/eval', eval_test, total_steps)

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rom', help='path of the rom of the atari game', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument(
        '--algo',
        default='DQN',
        help=
        'DQN/DDQN/Dueling, represent DQN, double DQN, and dueling DQN respectively',
    )
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e7),
        help='maximum environmental steps of games')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='the step interval between two consecutive evaluations')

    args = parser.parse_args()
    main()
