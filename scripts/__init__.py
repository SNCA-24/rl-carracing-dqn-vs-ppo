"""
scripts package

Provides:
  - train_main, train_agent       (training pipeline)
  - evaluate_main, evaluate_dqn, evaluate_ppo
  - record_video_main, record_video
  - plot_metrics_main
  - preprocess_frame, create_state_stack, save_training_progress
"""

from .train import main as train_main, train_agent
from .evaluate import main as evaluate_main, evaluate_dqn, evaluate_ppo
from .record_video import main as record_video_main, record_video
from .plot_metrics import main as plot_metrics_main
from .utils import preprocess_frame, create_state_stack, save_training_progress

__all__ = [
    "train_main", "train_agent",
    "evaluate_main", "evaluate_dqn", "evaluate_ppo",
    "record_video_main", "record_video",
    "plot_metrics_main",
    "preprocess_frame", "create_state_stack", "save_training_progress"
]