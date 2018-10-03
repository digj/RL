import numpy as np
import matplotlib.pyplot as plt

#generates one 10-armed test bed with means picked from gaussian distribution with 0 mean and deviation 1
def generate_bandit(No_arms):
	q_star = np.random.normal(0,1,No_arms)#true expected rewards
	std_dev = np.ones(No_arms)            #standard deviation of each arm
	return q_star,std_dev


