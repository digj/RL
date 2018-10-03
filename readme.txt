Algorithm files contain all the algorithms.
eps_greedy.py-epsilon greedy
softmax.py-softmax
UCB1.py-UCB1
median_elimination_algo.py-Median Eliminaton Algorithm
median_elimination_quick_sort.py-quick select sort calculation of median

####FOR ENVIRON, OPTION AND INTRAOPTION#####
The environment for this task is the following grid world. The cells of the grid correspond
to the states of the environment. There are four rooms numbered: 1, 2, 3 and 4. The room
number 1 is the one with ‘HALLWAYS’ written in it. The second room is on the right to
the first one, and so on. G1 and G2 are the goal states. States G1 and G2 give a reward of
+1, transitions to all the other states give a reward of 0. The discount factor is taken to be
γ = 0.9.There are four primitive actions: up, down, left and right. With probability 2/3, the actions
cause the agent to move one cell in the corresponding direction. With probability 1/3, the
agent instead moves in the other directions, each one with a probability 1/9. If the agent
hits the wall, it remains in the same state.
There are 8 multi-step options : each one leading to a particular room’s hallway. A hallway
option’s policy finds the shortest path within the room to its target hallway. The termination
condition β(s) for each hallway option is zero for states s within the room and 1 for the states
outside the room, including the hallway states. The initiation step I comprises the states
within the room plus the non-target hallway state leading to the room.
