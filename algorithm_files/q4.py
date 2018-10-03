import numpy as np
import matplotlib.pyplot as plt
import timeit


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

#Median elimination implemebtation
#q_star: actual reward to draw sample
#std_dev:Actual std deviation to draw sample
#eps1: epsilon
#delt1:delta
def median_elimination(q_star, std_dev, eps1, delt1):
	eps = eps1/4.0
	delt = delt1/2.0
	q = np.zeros(No_arms)#average reward of each arm
	n = np.zeros(No_arms)#number of times each arm is pulled
	arms = np.array(range(No_arms))
	while(1) :#iterate till we find epsilon optimal arm
		itertations = int((2.0/(eps)**2)*np.log(3.0/delt))#number of iterations at each round
		for i in range(itertations):
			n[arms] = n[arms]+1
			reward = draw_sample(q_star[arms],std_dev[arms])
			q[arms] = q[arms] + (1.0/(n[arms]))*(reward - q[arms])#update rule
		median = np.median(q[arms])#finding median
		arms = np.where(q > median)[0]#eliminating arms or finding arms > median
		eps = (3.0/4.0)*eps#updating epsilon
		delt = delt/2.0#updating delta
		if arms.shape[0] == 1:
			final_arm = arms
			break;
	return q[final_arm], final_arm

#calls MEA algorithm
#eps: epsilon
#delt:delta
#No_arms: number of arms
#no_bandits: no. ofbandits
def call_mea(no_bandits,No_arms,eps,delt):
	q_est = np.zeros(no_bandits)
	q_star_max = np.zeros(no_bandits)
	for b in range(no_bandits):
		print b
		q_star,std_dev = generate_bandit(No_arms)
		q_star_max[b] = max(q_star)
		q = np.zeros(No_arms)
		n = np.zeros(No_arms)
		q_est[b], final_arm = median_elimination(q_star,std_dev,eps,delt)
	return final_arm, q_est,q_star_max 

No_arms = 10
no_bandits = 100
eps = 0.1
delt = 0.1
arm, q_est, q_star_max  = call_mea(no_bandits,No_arms, eps, delt)
fig, ax = plt.subplots()
ax.set_title('For Eps = '+str(eps)+'delt = '+str(delt))
ax.set_xlabel('Bandit Number')
ax.set_ylabel('True reward - Estimated reward')
ax.plot(range(no_bandits),q_star_max-q_est, label = 'Difference')
ax.plot(range(no_bandits),np.ones(no_bandits)*(-eps), label = 'Lower bound')
ax.plot(range(no_bandits),np.ones(no_bandits)*eps, label = 'Upper bound')
legend = ax.legend(loc='lower right', shadow=True)
plt.show()

