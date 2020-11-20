import pickle
import matplotlib.pyplot as plt

with open('/Users/Anirudh/multiagent-particle-envs/results/maddpg_simple/test_rewards.pkl', 'rb') as f:
    rewarddata = pickle.load(f)

times = [1000*(i+1) for i in range(10)]

plt.plot(times,rewarddata)
plt.xlabel('# episodes trained')
plt.ylabel('average reward value')
plt.title('Learning curve Task 1.2')
plt.show()