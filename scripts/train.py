#!/usr/bin/env python3

"""
Train script for CarRacing RL benchmarking.
Supports DQN, DoubleDQN, DuelingDQN, PERDQN, and PPO.
Loads hyperparameters from config.yaml.
"""
import os
import time
import yaml
import argparse
import random
import numpy as np
import logging

# Import environment wrapper
from envs.carracing_wrapper import CarRacingEnv
# Import DQN variants
from algos.dqn_base import DQNAgent
from algos.double_dqn import DoubleDQNAgent
from algos.dueling_dqn import DuelingDQNAgent
from algos.per_dqn import PERDQNAgent
# Import PPO wrapper
from algos.ppo_loader import PPOAgentWrapper
# Import training helper
from scripts.utils import create_state_stack, save_training_progress

# Set up logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def train_agent(agent, agent_name, env, state_shape, config, model_dir, log_dir):
    """Train a DQN‑family agent."""
    # Unpack state dimensions
    height, width, stack_size = state_shape
    # Extract configs
    dqn_cfg = config['dqn']
    train_cfg = config['training']
    episodes = train_cfg['num_episodes']
    max_steps = train_cfg['max_steps']
    save_freq = train_cfg['save_freq']
    replay_freq = train_cfg['replay_frequency']

    # Tracking
    rewards, lengths, epsilons = [], [], []

    for ep in range(episodes):
        reset_out = env.reset()
        # Unpack observation from (obs, info) tuple if necessary
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        frames, stacked = create_state_stack(None, state, True, stack_size, width, height)
        ep_reward = 0
        ep_step = 0
        done = False
        truncated = False

        while not (done or truncated) and ep_step < max_steps:
            # Select and execute action
            action = agent.act(stacked)
            next_state, reward, done, truncated, _ = env.step(action)
            frames, stacked_next = create_state_stack(frames, next_state, False, stack_size, width, height)
            # Store and train
            agent.remember(stacked, action, reward, stacked_next, done or truncated)
            if ep_step % replay_freq == 0:
                agent.replay()
            stacked = stacked_next
            ep_reward += reward
            ep_step += 1

        # End of episode
        logger.info(f"{agent_name} Ep {ep+1}/{episodes} - Reward: {ep_reward:.2f} Steps: {ep_step} Eps: {getattr(agent,'epsilon',0):.4f}")
        rewards.append(ep_reward)
        lengths.append(ep_step)
        epsilons.append(getattr(agent,'epsilon',0))

        # Checkpoint
        if (ep+1) % save_freq == 0 or (ep+1)==episodes:
            ckpt = os.path.join(model_dir, f"{agent_name}_ep{ep+1}.h5")
            agent.save_model(ckpt)
            logger.info(f"Saved model: {ckpt}")
            save_training_progress(rewards, lengths, epsilons,
                                   os.path.join(log_dir, f"{agent_name}_ep{ep+1}"))

    # Final save
    final_path = os.path.join(model_dir, f"{agent_name}_final.h5")
    agent.save_model(final_path)
    logger.info(f"Final model saved: {final_path}")
    return {'rewards': rewards, 'lengths': lengths, 'epsilons': epsilons}


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on CarRacing-v2")
    parser.add_argument('--algo', type=str, required=True,
                        choices=['DQN','DoubleDQN','DuelingDQN','PERDQN','PPO'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    # Load config
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(args.model_dir, f"{args.algo}_{timestamp}")
    log_dir = os.path.join(args.log_dir, f"{args.algo}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    env_cfg = config['env']
    env = CarRacingEnv(render_mode=env_cfg['render_mode'])
    state_shape = (env_cfg['height'], env_cfg['width'], env_cfg['frame_stack'])

    # Instantiate and train
    if args.algo in ['DQN','DoubleDQN','DuelingDQN','PERDQN']:
        # Map algos
        cls_map = {
            'DQN': DQNAgent,
            'DoubleDQN': DoubleDQNAgent,
            'DuelingDQN': DuelingDQNAgent,
            'PERDQN': PERDQNAgent
        }
        AgentClass = cls_map[args.algo]
        # Pass additional PER config if needed
        if args.algo == 'PERDQN':
            agent = AgentClass(state_shape, env.action_space.n,
                               config['dqn'], config['per_dqn'])
        else:
            agent = AgentClass(state_shape, env.action_space.n, config['dqn'])
        train_metrics = train_agent(agent, args.algo, env, state_shape,
                                    config, model_dir, log_dir)
        print('Training complete:', train_metrics)

    else:  # PPO
        ppo_cfg = config['ppo']
        ppo_agent = PPOAgentWrapper(ppo_cfg, model_dir)
        timesteps = config['training'].get('ppo_timesteps', 100000)
        ppo_agent.learn(total_timesteps=timesteps)
        ppo_path = os.path.join(model_dir, 'PPO_final.zip')
        ppo_agent.save(ppo_path)
        print(f"PPO training complete. Model saved to {ppo_path}")

    env.close()

if __name__=='__main__':
    main()
