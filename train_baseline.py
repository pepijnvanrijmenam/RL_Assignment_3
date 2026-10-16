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
    Baseline strategy:
    Always have the same pole length of 0.5.
    """
    return 0.5


def main():
    print("Training baseline")
    for seed in SEEDS:
        train(
            get_pole_length_fn=get_pole_length,
            seed=seed,
            strategy_name="baseline",
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
    print("\nBaseline training finished")


if __name__ == "__main__":
    main()
