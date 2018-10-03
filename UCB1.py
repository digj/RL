import numpy as np
import matplotlib.pyplot as plt

#generates one 10-armed test bed with means picked from gaussian distribution with 0 mean and deviation 1
def generate_bandit(No_arms):
	q_star = np.random.normal(0,1,No_arms)#true expected rewards
	std_dev = np.ones(No_arms)            #standard deviation of each arm
	return q_star,std_dev
#draws a sample
#q_star: true mean of the arm to be pulled
#std_dev : true standard distribution of the arm to be pulled
def draw_sample(q_star, std_dev):
	reward = np.random.normal(q_star,std_dev)
	return reward
#Chooses arm depending
#q: average rewards till that time
#t: time step
#n: array specifying no. of times each arm is taken
#c: confidence 
def choose_arm(q, t, n, c):
	arm = np.argmax(q + c*(np.log(t)/n)**0.5)
	return arm

#UCB1 Algorithm
#No_arms: number of arms
#c: confidence 
def ucb1(No_arms,c):
	avg_reward = np.zeros((2000,1000))#to record avg rewards
	optimal_action_counts = np.zeros((2000,1000))#to record optimal action counts
	for b in range(2000):
		print b
		q_star,std_dev = generate_bandit(No_arms)#creates a bandit problem
		q = np.zeros(No_arms)#average reward of each arm
		n = np.zeros(No_arms)#number of times each arm is pulled
		q = draw_sample(q_star,std_dev)
		n = n+1
		for t in range(1000):
			arm = int(choose_arm(q, t, n, c))#chooses arm
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])#draw a sample
			q[arm] = q[arm] + (1.0/(n[arm]))*(reward - q[arm])#update rule for average rewards
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1
	optimal_action_counts = optimal_action_counts.mean(axis = 0)#counts optimal actions when taken
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts

No_arms = 1000#number of arms
avg_reward_2, optimal_action_counts_2 = ucb1(No_arms,2)# Implements UCB1
avg_reward_1, optimal_action_counts_1 = ucb1(No_arms,1)
avg_reward_10, optimal_action_counts_10 = ucb1(No_arms,10)
fig, ax = plt.subplots()
ax.set_title('Reward evolution')
ax.set_xlabel('iterations')
ax.set_ylabel('Average reward')
ax.plot(range(1000),avg_reward_2, label='c = 2')
ax.plot(range(1000),avg_reward_1,label = 'c = 1')
ax.plot(range(1000),avg_reward_10,label = 'c = 10')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
fig, ax = plt.subplots()
ax.set_title('Optimal actions %')
ax.set_xlabel('iterations')
ax.set_ylabel('optimal actions taken in %')
ax.plot(range(1000),100*optimal_action_counts_2, label='c = 2')
ax.plot(range(1000),100*optimal_action_counts_1,label = 'c = 1')
ax.plot(range(1000),100*optimal_action_counts_10,label = 'c = 10')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()

