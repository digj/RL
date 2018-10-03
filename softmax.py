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

#Chooses arm depending on whether to explore or exploit
#temp: temperature
#No_arms: number of arms per bandit
#q: estimated rewards till that time
def choose_arm(q,temp,No_arms):
	probs = np.exp(q/temp)/np.sum(np.exp(q/temp))
	arm = np.random.choice(np.arange(0,No_arms,1),replace = True, p = probs)
	return arm

#Softmax selection algorithm
#temp:temperature
#No_arms: number of arms per bandit
def softmax_choice(temp,No_arms):
	avg_reward = np.zeros((2000,1000))#to record avg rewards
	optimal_action_counts = np.zeros((2000,1000))#to record optimal action counts
	for b in range(2000):
		q_star,std_dev = generate_bandit(No_arms))#creates a bandit problem
		q = np.zeros(No_arms)#average reward of each arm
		n = np.zeros(No_arms)#number of times each arm is pulled
		for t in range(1000):
			arm = int(choose_arm(q,temp,No_arms))#decide whether to explore or exploit
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])#draw a sample
			q[arm] = q[arm] + (1/(n[arm]))*(reward - q[arm])#update rule for average rewards
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1#counts optimal actions when taken
	optimal_action_counts = optimal_action_counts.mean(axis = 0)
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts

No_arms = 1000#number of arms each bandit has
avg_reward_0, optimal_action_counts_0 = softmax_choice(0.01,No_arms)# Implements softmac
avg_reward_001,optimal_action_counts_001 =  softmax_choice(0.1,No_arms)
avg_reward_01,optimal_action_counts_01 = softmax_choice(1,No_arms)
fig, ax = plt.subplots()
ax.set_title('Reward evolution')
ax.set_xlabel('iterations')
ax.set_ylabel('Average reward')
ax.plot(range(1000),avg_reward_0, label='temp = 0.01')
ax.plot(range(1000),avg_reward_001,label = 'temp = 0.1')
ax.plot(range(1000),avg_reward_01,label = 'temp = 1')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
fig, ax = plt.subplots()
ax.set_title('Optimal actions %')
ax.set_xlabel('iterations')
ax.set_ylabel('optimal actions taken in %')
ax.plot(range(1000),100*optimal_action_counts_0, label='temp = 0.01')
ax.plot(range(1000),100*optimal_action_counts_001,label = 'temp = 0.1')
ax.plot(range(1000),100*optimal_action_counts_01,label = 'temp = 1')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
