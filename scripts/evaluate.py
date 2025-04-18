#!/usr/bin/env python3
"""
Evaluate a trained agent on CarRacing-v2.
Supports DQN variants and PPO.
Logs metrics and saves to JSON.
"""
import os
import yaml
import argparse
import numpy as np
import json

from envs.carracing_wrapper import CarRacingEnv
from algos.dqn_base import DQNAgent
from algos.double_dqn import DoubleDQNAgent
from algos.dueling_dqn import DuelingDQNAgent
from algos.per_dqn import PERDQNAgent
from algos.ppo_loader import PPOAgentWrapper
from scripts.utils import create_state_stack


def evaluate_dqn(agent_class, model_path, env, state_shape, eval_cfg, dqn_cfg=None, per_cfg=None):
    # unpack dimensions
    height, width, stack_size = state_shape

    # Instantiate appropriate DQN agent
    if agent_class == DQNAgent:
        agent = DQNAgent(state_shape, env.action_space.n, dqn_cfg)
    elif agent_class == DoubleDQNAgent:
        agent = DoubleDQNAgent(state_shape, env.action_space.n, dqn_cfg)
    elif agent_class == DuelingDQNAgent:
        agent = DuelingDQNAgent(state_shape, env.action_space.n, dqn_cfg)
    elif agent_class == PERDQNAgent:
        agent = PERDQNAgent(state_shape, env.action_space.n, dqn_cfg, per_cfg)
    else:
        raise ValueError(f"Unsupported DQN agent: {agent_class}")

    # Load weights and disable exploration
    agent.load_model(model_path)
    agent.epsilon = 0.0

    rewards, lengths, tiles = [], [], []
    episodes = eval_cfg['eval_episodes']
    max_steps = eval_cfg['eval_max_steps']

    for ep in range(episodes):
        obs, _ = env.reset()
        # initial frame stack
        frames, state = create_state_stack(None, obs, True,
                                           stack_size, width, height)
        ep_reward = 0
        ep_len = 0
        ep_tiles = 0
        done = False
        truncated = False

        while not (done or truncated) and ep_len < max_steps:
            a = agent.act(state, training=False)
            obs, r, done, truncated, info = env.step(a)
            frames, state = create_state_stack(frames, obs, False,
                                               stack_size, width, height)
            ep_reward += r
            ep_len += 1
            # Track tiles if available
            if 'tiles' in info:
                ep_tiles = info['tiles']
        rewards.append(ep_reward)
        lengths.append(ep_len)
        tiles.append(ep_tiles)
        print(f"Eval {ep+1}/{episodes}: Reward={ep_reward:.2f}, Steps={ep_len}")

    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'all_rewards': rewards,
        'all_lengths': lengths,
        'tiles': tiles
    }
    return results


def evaluate_ppo(model_path, env, eval_cfg):
    # Instantiate PPO wrapper to load model
    # config for PPO isn't needed here
    agent = PPOAgentWrapper(config=None)
    agent.load(model_path)

    rewards, lengths = [], []
    episodes = eval_cfg['eval_episodes']
    max_steps = eval_cfg['eval_max_steps']

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_len = 0
        done = False
        truncated = False

        while not (done or truncated) and ep_len < max_steps:
            a, _ = agent.model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(a)
            ep_reward += r
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"PPO Eval {ep+1}/{episodes}: Reward={ep_reward:.2f}, Steps={ep_len}")

    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'all_rewards': rewards,
        'all_lengths': lengths
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CarRacing agents')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['DQN','DoubleDQN','DuelingDQN','PERDQN','PPO'])
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights or PPO zip')
    parser.add_argument('--eval_episodes',    type=int, help='Override evaluation.eval_episodes')
    parser.add_argument('--eval_max_steps',   type=int, help='Override evaluation.eval_max_steps')
    args = parser.parse_args()

    # Load config
    cfg_path = os.environ.get("RL_CONFIG_PATH", "config.yaml")
    with open('cfg_path','r') as f:
        config = yaml.safe_load(f)
    # Apply any CLI evaluation overrides
    eval_cfg = config.get('evaluation', {})
    if args.eval_episodes is not None:
        eval_cfg['eval_episodes'] = args.eval_episodes
    if args.eval_max_steps is not None:
        eval_cfg['eval_max_steps'] = args.eval_max_steps
    config['evaluation'] = eval_cfg

    # Create env
    env_cfg = config['env']
    env = CarRacingEnv(render_mode='rgb_array')
    state_shape = (env_cfg['height'], env_cfg['width'], env_cfg['frame_stack'])

    # Evaluate accordingly
    if args.algo in ['DQN','DoubleDQN','DuelingDQN','PERDQN']:
        cls_map = {
            'DQN': DQNAgent,
            'DoubleDQN': DoubleDQNAgent,
            'DuelingDQN': DuelingDQNAgent,
            'PERDQN': PERDQNAgent
        }
        results = evaluate_dqn(
            cls_map[args.algo],
            args.model_path,
            env,
            state_shape,
            config['evaluation'],
            config['dqn'],
            config.get('per_dqn')
        )
    else:
        results = evaluate_ppo(
            args.model_path,
            env,
            config['evaluation']
        )

    # Save results
    save_path = f"eval_{args.algo}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {save_path}")

    env.close()

if __name__=='__main__':
    main()
