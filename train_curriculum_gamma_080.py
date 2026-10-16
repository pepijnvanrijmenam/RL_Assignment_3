import numpy as np
from shared_functions import train, SEEDS

# Hyperparameters
GAMMA = 0.80
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
    Curriculum gama 0.80 strategy:
    the same as the curriculum strategy, but with gamma=0.80
    """
    center = 1.1
    min_half_width = 0.1  # gives [1.0, 1.2]
    max_half_width = 0.7  # gives [0.4, 1.8]
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
    print("Training Curriculum gamma 0.80")
    for seed in SEEDS:
        train(
            get_pole_length_fn=get_pole_length,
            seed=seed,
            strategy_name="curriculum_g_080",
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
    print("\nCurriculum gamma 0.80 training finished")


if __name__ == "__main__":
    main()
