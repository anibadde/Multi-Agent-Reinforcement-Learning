import pickle
import matplotlib.pyplot as plt 

with open('/Users/Anirudh/multiagent-particle-envs/results/maddpg_uav/test_agrewards.pkl', 'rb') as f:
    reward_data = pickle.load(f)

times = [1000*(i+1) for i in range(5)]
poacher = []
ranger = []
uav = []

for i in range(len(reward_data)):
    if i%3 == 0:
        poacher.append(reward_data[i])
    elif i%3 == 1:
        ranger.append(reward_data[i])
    else:
        uav.append(reward_data[i])

plt.plot(times, poacher, color='blue', label = 'poacher')
plt.plot(times, ranger, color='red', label='ranger')
plt.plot(times, uav, color='red', label='uav')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend()
plt.show()

#print(reward_data)