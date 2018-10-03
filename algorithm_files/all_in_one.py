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

def choose_arm_sm(q,temp,No_arms):
	probs = np.exp(q/temp)/np.sum(np.exp(q/temp))
	arm = np.random.choice(np.arange(0,No_arms,1),replace = True, p = probs)
	return arm

def choose_arm_eps(explore,q,No_arms):
	if explore:
		arm = np.random.randint(0,No_arms,size = 1)
	else:
		arm = np.argmax(q)
	return arm

def choose_arm_ucb(q, t, n, c):
	arm = np.argmax(q + c*(np.log(t)/n)**0.5)
	return arm

def explore_or_exploit(eps):
	explore = np.random.binomial(1,eps,1)
	return explore

def softmax_choice(temp,No_arms):
	avg_reward = np.zeros((2000,1000))
	optimal_action_counts = np.zeros((2000,1000))
	for b in range(2000):
		print b
		q_star,std_dev = generate_bandit(No_arms)
		q = np.zeros(No_arms)
		n = np.zeros(No_arms)
		for t in range(1000):
			arm = int(choose_arm_sm(q,temp,No_arms))
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])
			q[arm] = q[arm] + (1/(n[arm]))*(reward - q[arm])
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1
	optimal_action_counts = optimal_action_counts.mean(axis = 0)
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts

def eps_greedy(eps,No_arms):
	avg_reward = np.zeros((2000,1000))
	optimal_action_counts = np.zeros((2000,1000))
	for b in range(2000):
		print b
		q_star,std_dev = generate_bandit(No_arms)
		q = np.zeros(No_arms)
		n = np.zeros(No_arms)
		for t in range(1000):
			exp_or_exp = explore_or_exploit(eps)
			arm = int(choose_arm_eps(exp_or_exp, q,No_arms))
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])
			q[arm] = q[arm] + (1.0/(n[arm]))*(reward - q[arm])
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1
	optimal_action_counts = optimal_action_counts.mean(axis = 0)
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts

def ucb1(No_arms,c):
	avg_reward = np.zeros((2000,1000))
	optimal_action_counts = np.zeros((2000,1000))
	for b in range(2000):
		print b
		q_star,std_dev = generate_bandit(No_arms)
		q = np.zeros(No_arms)
		n = np.zeros(No_arms)
		q = draw_sample(q_star,std_dev)
		n = n+1
		for t in range(1000):
			arm = int(choose_arm_ucb(q, t, n, c))
			n[arm] = n[arm] + 1
			reward = draw_sample(q_star[arm],std_dev[arm])
			q[arm] = q[arm] + (1.0/(n[arm]))*(reward - q[arm])
			avg_reward[b,t] = q[arm]
			if arm ==  np.argmax(q_star):
				optimal_action_counts[b,t] = optimal_action_counts[b,t] + 1
	optimal_action_counts = optimal_action_counts.mean(axis = 0)#counts optimal actions when taken
	avg_reward = avg_reward.mean(0)
	return avg_reward, optimal_action_counts


No_arms = 1000
avg_reward_0, optimal_action_counts_0 = eps_greedy(0.01,No_arms)
avg_reward_001,optimal_action_counts_001 =  softmax_choice(0.1,No_arms)
avg_reward_01,optimal_action_counts_01 = ucb1(No_arms,0.5)
fig, ax = plt.subplots()
ax.set_title('Reward evolution')
ax.set_xlabel('iterations')
ax.set_ylabel('Average reward')
ax.plot(range(1000),avg_reward_0, label='eps = 0.01')
ax.plot(range(1000),avg_reward_001,label = 'temp = 0.1')
ax.plot(range(1000),avg_reward_01,label = 'c = 0.5')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
fig, ax = plt.subplots()
ax.set_title('Optimal actions %')
ax.set_xlabel('iterations')
ax.set_ylabel('optimal actions taken in %')
ax.plot(range(1000),100*optimal_action_counts_0, label='eps = 0.01')
ax.plot(range(1000),100*optimal_action_counts_001,label = 'temp = 0.1')
ax.plot(range(1000),100*optimal_action_counts_01,label = 'c = 0.5')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
