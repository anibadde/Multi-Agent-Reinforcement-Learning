import csv
import matplotlib.pyplot as plt 
import numpy as np
import math

reward_mean = []
steps = []

#This plot uses the average in progress.csv

with open('/Users/Anirudh/multiagent-particle-envs/results/ppo_atari/progress.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['eprewmean'] == 'nan':
            reward_mean.append(-20.30)
        else:
            reward_mean.append(float(row['eprewmean']))
        steps.append(float(row['misc/total_timesteps']))

print(len(reward_mean))

fig = plt.plot(steps, reward_mean)
plt.xticks(np.arange(min(steps), max(steps)+1, 10000))
plt.yticks(np.arange(min(reward_mean), max(reward_mean)+1, 5.0))
plt.xlabel('# steps')
plt.ylabel('epsiode reward')
plt.title('Learning Curve Task 2')
plt.show()