import numpy as np
from shared_functions import train, SEEDS

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_BUFFER_SIZE = 1000
TARGET_UPDATE_FREQ = 500
N_EPISODES = 750
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = N_EPISODES / 2


def get_pole_length(episode, n_episodes):
    """
    Curriculum strategy:
    Starts with a narrow selection of heights from center [1.0, 1.2]
    Then linearly increases selection of height until reaching [0.4, 1.8]
    For each episode sample random pole length from the selection of height.
    """
    center = 1.1
    min_half_width = 0.1  # [1.0, 1.2]
    max_half_width = 0.7  # [0.4, 1.8]
    curriculum_episodes = 0.6 * n_episodes

    if episode < curriculum_episodes:
        progress = episode / curriculum_episodes
        half_width = min_half_width + progress * (max_half_width - min_half_width)
    else:
        half_width = max_half_width

    low = center - half_width
    high = center + half_width
    return float(np.random.uniform(low, high))


def main():
    print("Training curriculum")
    for seed in SEEDS:
        train(
            get_pole_length_fn=get_pole_length,
            seed=seed,
            strategy_name="curriculum",
            n_episodes=N_EPISODES,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            min_buffer_size=MIN_BUFFER_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay_episodes=EPS_DECAY_EPISODES,
        )
    print("\nCurriculum training finished")


if __name__ == "__main__":
    main()
