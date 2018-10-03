import numpy as np
import gym
import random as rd
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
gamma = 0.9
alpha = 0.1

##Defining options

def policy_option_0(state):#room1
    if state[0] == 5:
        action = UP
        return action
    if state[1]<4:
        action = RIGHT
        return action
    if state[1]==4 and state[0] < 2:
        action = DOWN
        return action
    if state[1]==4 and state[0] > 2:
        action = UP
        return action
    if state[1]==4 and state[0]==2:
        action = RIGHT
        return action

def policy_option_1(state):#room1
    if state[1]==5:
        action = LEFT
        return action
    if state[0]<4:
        action = DOWN
        return action
    if state[0]==4 and state[1]<1:
        action = RIGHT
        return action
    if state[0]==4 and state[1]>1:
        action = LEFT
        return action
    if state[0]==4 and state[1]==1:
        action = DOWN
        return action

def policy_option_4(state):#room3 stop@[2,-1]
    if state[0] == -1:
        return DOWN
    if state[1] >0:
        return LEFT
    if state[1]==0 and state[0]<2:
        return DOWN
    if state[1]==0 and state[0]>2:
        return UP
    if state[1]==0 and state[0]==2:
        return LEFT

def policy_option_5(state):#room 3 stop@[-1,2]
    if state[1] == -1:
        return RIGHT
    if state[0]>0:
        return UP
    if state[0]==0 and state[1] > 2:
        return LEFT
    if state[0]==0 and state[1] < 2:
        return RIGHT
    if state[0]==0 and state[1]==2:
        return UP

def policy_option_6(state):#room4
    # print(state)
    if state[0] == -1:
        return DOWN
    if state[1]<4:
        return RIGHT
    if state[1]==4 and state[0]< 3:
        return DOWN
    if state[1]==4 and state[0]>3:
        return UP
    if state[1]==4 and state[0]==3:
        return RIGHT

def policy_option_7(state):#room4
    if state[1]==5:
        return LEFT
    if state[0]>0:
        return UP
    if state[0]==0 and state[1]>1:
        return LEFT
    if state[0]==0 and state[1]<1:
        return RIGHT
    if state[0]==0 and state[1]==1:
        return UP

def policy_option_2(state):#room2 stop@[6,2]
    if state[1]==-1:
        return RIGHT
    if state[0]<5:
        return DOWN
    if state[0]==5 and state[1]<2:
        return RIGHT
    if state[0]==5 and state[1]>2:
        return LEFT
    if state[0]==5 and state[1]==2:
        return DOWN

def policy_option_3(state):#room2 stop@[2,-1]
    if state[0]==6:
        return UP
    if state[1]>0:
        return LEFT
    if state[1]==0 and state[0]<2:
        return DOWN
    if state[1]==0 and state[0]>2:
        return UP
    if state[1]==0 and state[0]==2:
        return LEFT

def is_action_avaibale(state,action):
    available = []
    if state[0] == 0 and action == policy_option_0(state[1]):
        available.append(0)
    if state[0] == 0 and action == policy_option_1(state[1]):
        available.append(1)
    if state[0] == 1 and action == policy_option_2(state[1]):
        available.append(2)
    if state[0] == 1 and action == policy_option_3(state[1]):
        available.append(3)
    if state[0] == 2 and action == policy_option_4(state[1]):
        available.append(4)
    if state[0] == 2 and action == policy_option_5(state[1]):
        available.append(5)
    if state[0] == 3 and action == policy_option_6(state[1]):
        available.append(6)
    if state[0] == 3 and action == policy_option_7(state[1]):
        available.append(7)
    if action == UP:
        available.append(8)
    if action == DOWN:
        available.append(9)
    if action == RIGHT:
        available.append(10)
    if action == LEFT:
        available.append(11)
    return np.array(available)


#executing the option selected
def execute_option(option,state,ob_enc,env,u_s1_o,q_s_o):
    reward = 0
    i = 0
    if option == 8:
        action = UP
        avail_options = is_action_avaibale(state, action)
        next_ob, rew, done, next_ob_enc = env.step(action)
        terminate = 1
        u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate* (np.amax(q_s_o[next_ob_enc, :]))
        q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
        i = 1
        return rew, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
    if option == 9:
        action = DOWN
        avail_options = is_action_avaibale(state, action)
        next_ob, rew, done, next_ob_enc = env.step(action)
        terminate = 1
        u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
        q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
        i = 1
        return rew, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
    if option == 10:
        action = RIGHT
        avail_options = is_action_avaibale(state, action)
        next_ob, rew, done, next_ob_enc = env.step(action)
        terminate = 1
        u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
        q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
        i = 1
        return rew, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
    if option == 11:
        action = LEFT
        avail_options = is_action_avaibale(state, action)
        next_ob, rew, done, next_ob_enc = env.step(action)
        terminate = 1
        u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
        q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
        i = 1
        return rew, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
    while True:
        terminate = 0
        if option == 0:#hall way to room 2
            if state[0] == 3:
                print("slipped")
                state = [0,[5,1]]
                action = policy_option_0(state[1])
            else:
                action = policy_option_0(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 0 and next_ob[1] == [2, 5]) or (next_ob[0] == 1 and next_ob[1] == [2, -1]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            print(next_ob)
            state = next_ob
            reward = reward + rew*gamma**(i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 1:#hall way to room 4
            if state[0] == 1:
                print("slipped")
                action = policy_option_3(state[1])
            else:
                action = policy_option_1(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 0 and next_ob[1] == [5, 1]) or (next_ob[0] == 3 and next_ob[1] == [-1, 1]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            print(next_ob)
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 2:#hallway to room 3
            if state[0] == 0:
                print("slipped")
                state = [1,[2,-1]]
                action = policy_option_0(state[1])
            else:
                action = policy_option_2(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 1 and next_ob[1] == [6, 2]) or (next_ob[0] == 2 and next_ob[1] == [-1,2]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 3:#hallway to room 1
            if state[0] == 2:
                print("slipped")
                action = policy_option_5(state[1])
            else:
                action = policy_option_3(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 1 and next_ob[1] == [2, -1]) or (next_ob[0] == 0 and next_ob[1] == [2, 5]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 4:#hallway to room 4
            if state[0]==1:
                print("slipped")
                state = [2,[-1,2]]
                action = policy_option_2(state[1])
            else:
                action = policy_option_4(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 2 and next_ob[1] == [2, -1]) or (next_ob[0] == 3 and next_ob[1] == [3, 5]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 5:#hallway to room 2
            if state[0] == 3:
                print("slipped")
                action = policy_option_6(state[1])
            else:
                action = policy_option_5(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if (next_ob[0] == 2 and next_ob[1] == [-1, 2]) or (next_ob[0] == 1 and next_ob[1] == [6,2]):
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 6:#hallway to room 3
            if state[0] == 0:
                print("slipped")
                action = policy_option_1(state[1])
            else:
                action = policy_option_6(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 3 and next_ob[1] == [3, 5]) or (next_ob[0] == 2 and next_ob[1] == [2, -1]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            print(next_ob)
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        if option == 7:#hallway t room 1
            if state[0] == 2:
                print("slipped")
                state = [3,[3,5]]
                action = policy_option_4(state[1])
            else:
                action = policy_option_7(state[1])
            print(action)
            avail_options = is_action_avaibale(state, action)
            next_ob, rew, done, next_ob_enc = env.step(action)
            if (next_ob[0] == 3 and next_ob[1] == [-1, 1]) or (next_ob[0] == 0 and next_ob[1] == [5, 1]):
                terminate = 1
            u_s1_o[next_ob_enc, avail_options] = (1 - terminate) * q_s_o[next_ob_enc, avail_options] + terminate * (np.amax(q_s_o[next_ob_enc, :]))
            q_s_o[ob_enc, avail_options] = q_s_o[ob_enc, avail_options] + alpha * (rew + (gamma) * u_s1_o[next_ob_enc, avail_options] - q_s_o[ob_enc, avail_options])
            if done:
                reward = reward + rew * gamma ** (i)
                return reward, next_ob_enc, next_ob, i, done, u_s1_o, q_s_o
            state = next_ob
            reward = reward + rew * gamma ** (i)
            if terminate == 1:
                print("hallway found")
                return reward, next_ob_enc, next_ob,i,done, u_s1_o, q_s_o
            continue
        i = i+1


##select the option
def gridworld_get_action(q,ob_enc,ob,eps):
    exp_or_exp = np.random.binomial(1, eps, 1)
    print("in room:", ob)
    if ob[0] == 0:#room1
        if ob == [0,[5,1]]:
            if exp_or_exp == 1:#exploit
                action = np.argmax(q[ob_enc, [0, 6,8, 9, 10, 11]])
                if action == 0:
                    return 0
                elif action == 1:
                    return 6
                else:
                    return (action+6)
            else:
                return rd.choice([8, 9, 10, 11, 0, 6])
        if ob == [0,[2,5]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [ 1,2, 8, 9, 10, 11]])
                if action == 0:
                    return 1
                elif action == 1:
                    return 2
                else:
                    return (action + 6)
            else:
                return rd.choice([ 8, 9, 10, 11,1,2])
        if exp_or_exp == 1:#exploit
            print("exploit")
            action = np.argmax(q[ob_enc, [0,1,8,9,10,11]])
            if action == 0:
                if q[ob_enc,0] == 0.0:
                    return rd.choice([0,1])
                return 0
            if action == 1:
                if q[ob_enc,1] == 0.0:
                    return rd.choice([0,1])
                return 1
            if action > 1:
                return (action+6)
            # if q[ob_enc,0]>q[ob_enc,1]:
            #     action = 0
            # else:
            #     action = 1
        else:
            print("exploring")
            action = rd.choice([0,1,8,9,10,11])
        return action
    if ob[0]==1:#room2
        if ob == [1,[2,-1]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [2 , 1, 8, 9, 10, 11]])
                if action == 0:
                    return 2
                elif action == 1:
                    return 1
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 2 , 1])
        if ob == [1,[6,2]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [3, 4, 8, 9, 10, 11]])
                if action == 0:
                    return 3
                elif action == 1:
                    return 4
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 3, 4])
        if exp_or_exp == 1:
            action = np.argmax(q[ob_enc, [2, 3, 8, 9, 10, 11]])
            if action == 0:
                if q[ob_enc,2] == 0.0:
                    return rd.choice([2,3])
                return 2
            if action == 1:
                if q[ob_enc,3] == 0.0:
                    return rd.choice([2,3])
                return 3
            if action > 1:
                return (action+6)
            # if q[ob_enc,2]>q[ob_enc,3]:
            #     action = 2
            # else:
            #     action = 3
        else:
            print("exploring")
            action = rd.choice([2,3,8,9,10,11])
        return action
    if ob[0]==2:#room3
        if ob == [2,[-1,2]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [4, 2, 8, 9, 10, 11]])
                if action == 0:
                    return 4
                elif action == 1:
                    return 2
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 4, 2])
        if ob == [2,[2,-1]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [5, 7,8, 9, 10, 11]])
                if action == 0:
                    return 5
                elif action == 1:
                    return 7
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 5, 7])
        if exp_or_exp == 1:
            action = np.argmax(q[ob_enc, [4, 5, 8, 9, 10, 11]])
            if action == 0:
                if q[ob_enc,4] == 0.0:
                    return rd.choice([4,5])
                return 4
            if action == 1:
                if q[ob_enc,5] == 0.0:
                    return rd.choice([4,5])
                return 5
            if action > 1:
                return (action + 6)
            # if q[ob_enc,4]>q[ob_enc,5]:
            #     action = 4
            # else:
            #     action = 5
        else:
            print("exploring")
            action = rd.choice([4,5,8,9,10,11])
        return action
    if ob[0]==3:#room4
        if ob == [3,[-1,1]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [6, 0, 8, 9, 10, 11]])
                if action == 0:
                    return 6
                elif action == 1:
                    return 0
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 6, 0])
        if ob == [3,[3,5]]:
            if exp_or_exp == 1:  # exploit
                action = np.argmax(q[ob_enc, [7, 5, 8, 9, 10, 11]])
                if action == 0:
                    return 7
                if action == 1:
                    return 5
                else:
                    return (action + 6)
            else:
                return rd.choice([8, 9, 10, 11, 7, 5])
        if exp_or_exp == 1:
            action = np.argmax(q[ob_enc, [6, 7, 8, 9, 10, 11]])
            if action == 0:
                if q[ob_enc,6] == 0.0:
                    return rd.choice([6,7])
                return 6
            if action == 1:
                if q[ob_enc,7] == 0.0:
                    return rd.choice([6,7])
                return 7
            if action > 1:
                return (action + 6)
        else:
            print("exploring")
            action = rd.choice([6,7,8,9,10,11])
        return action

def main():
    from environ import FourRooms
    env = gym.make('FourRooms-v0')
    action = 1
    hallway_coords = [[2, 5], [6, 2], [2, -1], [-1, 1]]
    # q_s_o = np.random.rand(106,12)
    q_s_o = np.zeros((106,12))
    u_s1_o = np.zeros((106,12))
    eps = 0.6
    i = 0
    steps = []
    while True:
        print("NEW EPISODE:****************************************************", i)
        i = i+1
        # if i>750:
        #     eps = 1
        if i%100 == 0:
            eps = eps + 0.05
            if eps > 0.99:
                eps = 0.99
        if i == 5000:
            break
        ob_enc,ob = env.reset()
        ob = list(ob)
        print(ob)
        done = False
        k = 0
        while not done:
            k = k+1
            option = gridworld_get_action(q_s_o,ob_enc,ob,eps)
            print("option: ",option)
            reward, next_ob_enc, next_ob,j, done, u_s1_o, q_s_o = execute_option(option, ob,ob_enc,env,u_s1_o,q_s_o)
            print("updated")
            ob = next_ob
            ob_enc = next_ob_enc
        steps.append(k)

    print(q_s_o)
    plt.plot(range(len(steps)), steps)
    plt.show()
    plt.plot(range(q_s_o.shape[0]), np.amax(q_s_o, axis=1))
    plt.show()
    print("done")
    q = np.amax(q_s_o, axis=1)
    q_0 = np.zeros((6, 6))
    q_1 = np.zeros((7, 6))
    q_2 = np.zeros((5, 6))
    q_3 = np.zeros((6, 6))
    size = 22
    # plotting results
    for i in range(25):
        q_0[int(np.floor(i / 5)), i % 5] = q[i]
    q_0[2, 5] = q[25]
    s = [(size * q_0.flatten()[i]) ** 2 for i in range(6 * 6)]
    print(s)
    x0, y0 = np.meshgrid(np.array(range(6)), np.array(range(0, -6, -1)))
    plt.title("room1")
    plt.scatter(x0.flatten(), y0.flatten(), s=s)
    plt.grid()
    plt.show()
    for i in range(30):
        q_1[int(np.floor(i / 5)), i % 5] = q[i + 26]
    q_1[6, 2] = q[56]
    s = [(size * q_1.flatten()[i]) ** 2 for i in range(7 * 6)]
    plt.title("room2")
    x1, y1 = np.meshgrid(np.array(range(6)), np.array(range(0, -7, -1)))
    plt.scatter(x1.flatten(), y1.flatten(), s=s)
    plt.grid()
    plt.show()
    for i in range(20):
        q_2[int(np.floor(i / 5)), 1 + i % 5] = q[i + 57]
    q_2[2, 0] = q[77]
    s = [(size * q_2.flatten()[i]) ** 2 for i in range(5 * 6)]
    x2, y2 = np.meshgrid(np.array(range(6)), np.array(range(0, -5, -1)))
    plt.title("room4")
    plt.scatter(x2.flatten(), y2.flatten(), s=s)
    plt.grid()
    plt.show()
    for i in range(25):
        q_3[int(1 + np.floor(i / 5)), i % 5] = q[i + 78]
    q_3[0, 1] = q[103]
    s = [(size * q_3.flatten()[i]) ** 2 for i in range(6 * 6)]
    x3, y3 = np.meshgrid(np.array(range(6)), np.array(range(1, -5, -1)))
    plt.title("room3")
    plt.scatter(x3.flatten(), y3.flatten(), s=s)
    # axarr[0, 0].set_title('room3')
    plt.grid()
    plt.show()
    # plt.grid()
    # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    # plt.show()


main()
