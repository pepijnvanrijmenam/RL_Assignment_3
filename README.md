# RL_Assignment_3
Group 10

RL VU Amsterdam (XB_0118) 16.10.2026

This project trains a DQN agent to balance poles of different lengths in CartPole environment. We compare training strategies and test how well the agents handle wind and increasing forces.

python version = 3.10

pip install -r requirements.txt

run train_baseline.py to train basline
run train_uniform.py to train uniform
run train_curriculum.py to train curriculum
run train_curriculum_gamma_080.py to train train_curriculum_gamma_080

after training, run eval.py to evaluate and test the trained model

after evaluation, run statistical_test.py to get statistical test data
