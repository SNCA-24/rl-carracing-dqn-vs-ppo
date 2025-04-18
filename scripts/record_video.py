#!/usr/bin/env python3
"""
Record a short video of a trained agent playing CarRacing-v2.
Supports DQN variants and PPO; saves to disk under a timestamped folder.
"""
import os
import yaml
import argparse
from datetime import datetime
import numpy as np
import moviepy
import logging
from gymnasium.error import DependencyNotInstalled

from envs.carracing_wrapper import CarRacingEnv
from algos.dqn_base import DQNAgent
from algos.double_dqn import DoubleDQNAgent
from algos.dueling_dqn import DuelingDQNAgent
from algos.per_dqn import PERDQNAgent
from algos.ppo_loader import PPOAgentWrapper
from scripts.utils import create_state_stack

logger = logging.getLogger(__name__)

def record_video(agent, agent_name, model_path, config, video_base_dir):
    # Load env config
    env_cfg = config['env']
    frame_stack = env_cfg['frame_stack']
    width, height = env_cfg['width'], env_cfg['height']

    # Prepare video folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_dir = os.path.join(video_base_dir, f"{agent_name}_{timestamp}")
    os.makedirs(video_dir, exist_ok=True)

    # Create underlying gym env and wrap for recording
    import gymnasium as gym
    raw_env = gym.make('CarRacing-v2', render_mode='rgb_array')
    try:
        video_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=agent_name.lower()
        )
    except (DependencyNotInstalled, ModuleNotFoundError) as e:
        logger.warning(f"Video recording unavailable: {e}. Proceeding without saving video.")
        wrapped_env = CarRacingEnv(render_mode='rgb_array')
    else:
        wrapped_env = CarRacingEnv(render_mode='rgb_array')
        wrapped_env.env = video_env

    # Load the trained model
    if isinstance(agent, PPOAgentWrapper):
        agent.load(model_path)
    else:
        agent.load_model(model_path)
        agent.epsilon = 0.0

    # Run one episode recording up to N steps
    eval_steps = config['evaluation']['video_max_steps']
    reset_out = wrapped_env.reset()
    # Unpack observation only, in case reset returns (obs, info)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    frames, state = create_state_stack(None, obs, True, frame_stack, width, height)
    total_reward, step = 0, 0
    done = False
    truncated = False
    while not (done or truncated) and step < eval_steps:
        action = agent.act(state, training=False)
        obs, reward, done, truncated, _ = wrapped_env.step(action)
        frames, state = create_state_stack(frames, obs, False, frame_stack, width, height)
        total_reward += reward
        step += 1

    wrapped_env.close()
    print(f"Recorded video to {video_dir}")
    print(f"Steps: {step}, Reward: {total_reward:.2f}")
    return video_dir, step, total_reward


def main():
    parser = argparse.ArgumentParser(description='Record video of trained agent')
    parser.add_argument('--algo', required=True, choices=['DQN','DoubleDQN','DuelingDQN','PERDQN','PPO'])
    parser.add_argument('--model_path', required=True, help='Path to model weights or PPO zip')
    parser.add_argument('--video_dir', default='videos', help='Base directory for recordings')
    args = parser.parse_args()

    # Load config
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    # Instantiate agent
    if args.algo in ['DQN','DoubleDQN','DuelingDQN']:
        cls_map = {
            'DQN': DQNAgent,
            'DoubleDQN': DoubleDQNAgent,
            'DuelingDQN': DuelingDQNAgent
        }
        agent = cls_map[args.algo](
            (config['env']['height'], config['env']['width'], config['env']['frame_stack']),
            len(CarRacingEnv().actions),
            config['dqn']
        )
    elif args.algo == 'PERDQN':
        agent = PERDQNAgent(
            (config['env']['height'], config['env']['width'], config['env']['frame_stack']),
            len(CarRacingEnv().actions),
            config['dqn'], config['per_dqn']
        )
    else:  # PPO
        agent = PPOAgentWrapper(config['ppo'], model_dir=None)

    # Record
    record_video(agent, args.algo, args.model_path, config, args.video_dir)

if __name__ == '__main__':
    main()
