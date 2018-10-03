import numpy as np
import matplotlib.pyplot as plt
import random
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
#taken from internet and modified
#Implements Quickselect sorting
def Partition(a):
  """
  Usage: (left,pivot,right) = Partition(array)
  Partitions an array around a randomly chosen pivot such that 
  left elements <= pivot <= right elements.
  Running time: O(n)
  """
  ## Base cases
  if isinstance(a,float):
  	a = np.array([a])
  if len(a)==1: 
    return([],a[0],[])
  if len(a)==2: 
    if a[0]<=a[1]:
      return([],a[0],a[1])
    else:
      return([],a[1],a[0])
  ## Choose a random pivot
  p = random.randint(0,len(a)-1)  ## the pivot index
  pivot = a[p]  ## the pivot value
  right = []    ## the right partition
  left = []     ## the left partition
  for i in range(len(a)):
    if not i == p:
      if a[i] > pivot:
        right.append(a[i])
      else:
        left.append(a[i])
  return(left, pivot, right)
  
def QuickSelect(a,k):
  """
  Usage: kth_smallest_element = QuickSelect(array,k)
  Finds the kth smallest element of an array in linear time.
  """
  (left,pivot,right) = Partition(a)
  if len(left)==k-1:   ## pivot is the kth smallest element
    result = pivot
  elif len(left)>k-1: ## the kth element is in the left partition
    result = QuickSelect(left,k)
  else:               ## the kth element is in the right partition
    result = QuickSelect(right,k-len(left)-1)
  return result

def medi_an(q):
	if len(q)%2 == 0:
		# print "even"
		a = QuickSelect(q,len(q)/2)
		b = QuickSelect(q,len(q)/2+1)
		return (a+b)/2.0
	if len(q)%2 == 1:
		# print "odd"
		a = QuickSelect(q,len(q)/2+1)
		return a

#Median elimination implemebtation
#q_star: actual reward to draw sample
#std_dev:Actual std deviation to draw sample
#eps1: epsilon
#delt1:delta
def median_elimination(q_star, std_dev, eps1, delt1):
	# avg_reward = np.zeros((2000,1000))
	# optimal_action_counts = np.zeros((2000,1000))
	eps = eps1/4.0
	delt = delt1/2.0
	q = np.zeros(No_arms)
	n = np.zeros(No_arms)
	arms = np.array(range(No_arms))
	while(1) :
		itertations = int((2.0/(eps)**2)*np.log(3.0/delt))
		for i in range(itertations):
			n[arms] = n[arms]+1
			reward = draw_sample(q_star[arms],std_dev[arms])
			q[arms] = q[arms] + (1.0/(n[arms]))*(reward - q[arms])
		median = medi_an(q[arms])#calling quick select sorting algo
		arms = np.where(q > median)[0]
		eps = (3.0/4.0)*eps
		delt = delt/2.0
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
	# avg_reward = np.zeros((1,1000))
	# optimal_action_counts = np.zeros((1,1000))
	q_est = np.zeros(no_bandits)
	q_star_max = np.zeros(no_bandits)
	for b in range(no_bandits):
		print b
		q_star,std_dev = generate_bandit(No_arms)
		q_star_max[b] = max(q_star)
		q = np.zeros(No_arms)
		n = np.zeros(No_arms)
		t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
		q_est[b], final_arm = median_elimination(q_star,std_dev,eps,delt)
		print t.timeit()
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

