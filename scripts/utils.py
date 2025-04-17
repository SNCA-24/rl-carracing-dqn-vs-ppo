import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from collections import deque

def preprocess_frame(frame, width, height):
    """Convert frame to grayscale, resize, and normalize to [0,1] float."""
    # Convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(grayscale, (width, height), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = resized / 255.0
    return normalized


def create_state_stack(frames, state, is_new_episode, stack_size=4, width=84, height=84):
    """Maintain a deque of the last stack_size processed frames and return batch-ready array."""
    processed = preprocess_frame(state, width, height)
    if is_new_episode or frames is None:
        frames = deque([processed] * stack_size, maxlen=stack_size)
    else:
        frames.append(processed)
    stacked = np.stack(frames, axis=2)
    return frames, np.expand_dims(stacked, axis=0)


def save_training_progress(rewards, lengths, epsilons, filepath_prefix):
    """Save rewards, lengths, epsilons arrays and plot into progress image."""
    # Ensure output dir exists
    os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
    # Save arrays
    np.save(f"{filepath_prefix}_rewards.npy", np.array(rewards))
    np.save(f"{filepath_prefix}_lengths.npy", np.array(lengths))
    np.save(f"{filepath_prefix}_epsilons.npy", np.array(epsilons))

    # Create plot
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1,3,2)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.subplot(1,3,3)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(f"{filepath_prefix}_progress.png")
    plt.close()
