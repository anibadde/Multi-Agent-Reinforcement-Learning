import pickle
import matplotlib.pyplot as plt 

with open('/Users/Anirudh/multiagent-particle-envs/results/maddpg_uav/eval.pkl', 'rb') as f:
    reward_data = pickle.load(f)

print(reward_data)