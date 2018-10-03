import numpy as np
import matplotlib.pyplot as plt

#generates one 10-armed test bed with means picked from gaussian distribution with 0 mean and deviation 1
def generate_bandit(No_arms):
	q_star = np.random.normal(0,1,No_arms)#true expected rewards
	std_dev = np.ones(No_arms)            #standard deviation of each arm
	return q_star,std_dev

#decides whether to explore or exploit
#eps: epsilon
def explore_or_exploit(eps):
	explore = np.random.binomial(1,eps,1)#bernoulli random variable generator
	return explore

#draws a sample
#q_star: true mean of the arm to be pulled
#std_dev : true standard distribution of the arm to be pulled
def draw_sample(q_star, std_dev):
	reward = np.random.normal(q_star,std_dev)
	return reward

#Chooses arm depending on whether to explore or exploit
#explore:If true then explore else exploit
#q: estimated rewards till that time
def choose_arm(explore,q,No_arms):
	if explore:
		arm = np.random.randint(0,No_arms,size = 1)
	else:
		arm = np.argmax(q)
	return arm

#epsilon greedy algorithm
#eps:Epsilon
#No_arms: number of arms each bandit must have
def eps_greedy(eps,No_arms):
	avg_reward = np.zeros((2000,1000))#to record avg rewards
	optimal_action_counts = np.zeros((2000,1000))#to record optimal action counts
	for b in range(2000):
		print b
		q_star,std_dev = generate_bandit(No_arms)#creates a bandit problem
		q = np.zeros(No_arms)#average reward of each arm
		n = np.zeros(No_arms)#number of times each arm is pulled
		for t in range(1000):
			exp_or_exp = explore_or_exploit(eps)#decide whether to explore or exploit
			arm = int(choose_arm(exp_or_exp, q,No_arms))#chooses arm
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])#draw a sample
			q[arm] = q[arm] + (1.0/(n[arm]))*(reward - q[arm])#update rule for average rewards
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1#counts optimal actions when taken
	optimal_action_counts = optimal_action_counts.mean(axis = 0)
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts

No_arms = 10#number of arms each bandit has
avg_reward_0, optimal_action_counts_0 = eps_greedy(0.1,No_arms)# Implements epsilon greedy
avg_reward_001,optimal_action_counts_001 =  eps_greedy(0.01,No_arms)
avg_reward_01,optimal_action_counts_01 = eps_greedy(0,No_arms)
#plotting
fig, ax = plt.subplots()
ax.set_title('Reward evolution')
ax.set_xlabel('iterations')
ax.set_ylabel('Average reward')
ax.plot(range(1000),avg_reward_0, label='eps = 0.1')
ax.plot(range(1000),avg_reward_001,label = 'eps = 0.01')
ax.plot(range(1000),avg_reward_01,label = 'eps = 0')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
fig, ax = plt.subplots()
ax.set_title('Optimal actions %')
ax.set_xlabel('iterations')
ax.set_ylabel('optimal actions taken in %')
ax.plot(range(1000),100*optimal_action_counts_0, label='eps = 0.1')
ax.plot(range(1000),100*optimal_action_counts_001,label = 'eps = 0.01')
ax.plot(range(1000),100*optimal_action_counts_01,label = 'eps = 0')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()