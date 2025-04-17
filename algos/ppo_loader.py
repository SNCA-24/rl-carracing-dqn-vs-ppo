import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.carracing_wrapper import CarRacingEnv


def make_ppo_model(config, model_dir=None):
    """
    Create and return a PPO model configured via `config` (dict from config.yaml).
    - config: dict containing ppo hyperparameters
    - model_dir: optional path for tensorboard logs
    """
    # Create vectorized Gym environment
    def _env_fn():
        return CarRacingEnv(render_mode=None)
    vec_env = DummyVecEnv([_env_fn])

    # Set up tensorboard log directory if provided
    tb_log = None
    if model_dir:
        tb_log = os.path.join(model_dir, "ppo_tensorboard")
        os.makedirs(tb_log, exist_ok=True)

    # Instantiate PPO from stable_baselines3
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        clip_range=config["clip_range"],
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        tensorboard_log=tb_log,
        verbose=1
    )
    return model

class PPOAgentWrapper:
    """
    A thin wrapper around SB3 PPO to unify interface with DQNAgents.
    """
    def __init__(self, config, model_dir=None):
        self.model = make_ppo_model(config, model_dir)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, state, deterministic=True):
        # SB3 expects state with batch dimension
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save_model(self, filepath):
        """Save the PPO model, matching DQNAgent API."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load the PPO model, matching DQNAgent API."""
        self.model = PPO.load(filepath)
        self.model.set_env(self.model.get_env())

    def act(self, state, training=True):
        """Return action for a given state, matching DQNAgent API."""
        # deterministic during evaluation, randomization is inside `learn`
        return self.predict(state, deterministic=not training)

    def replay(self):
        """Dummy replay to match DQNAgent interface; PPO uses .learn() instead."""
        pass
