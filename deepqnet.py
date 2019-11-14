import numpy as np
import gym
#import gym_gridworld

import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import Policy
from rl.memory import SequentialMemory

def getKey(item):
    return item[0]

class CausalPolicy(Policy):
        #actions MUST BE ORDERED SAME WAY AS IN ENVIRONMENT
    def __init__(self, CGraph, ANodes, R, Epsilon): #Anodes is list of indices of action nodes in graph, R is index of reward node
        this.cgraph = CGraph #graph; adjacency table
        this.anodes = ANodes #set of indices of (directly?) intervene-able nodes
        this.epsilon = Epsilon
        for j in range(0, length(CGraph[0])):
            suminc = 0
            for i in range(0, length(CGraph)):
                suminc += CGraph[i][j]
            for i in range(0, length(CGraph)):
                CGraph[i][j] /= suminc

        ActionProbs = self.ProbTraverse(CGraph, R, ANodes)
        RActionProbs = self.RandActProb(ActionProbs, self.epsilon)
        self.actionProbs = ActionProbs

    def RandActProb(ActionProbs, epsilon): #epsilon randomness as implemented in random walk
        H = [1 / len(ActionProbs) for prob in ActionProbs]
        for i in range(0, H): 
            H[i] *= epsilon

        for i in range(0, len(ActionProbs)): 
            ActionProbs[i] *= (float(1) - epsilon)

        for i in range(0, len(ActionProbs)): 
            ActionProbs[i] += H[i]

        return ActionProbs






    def InvertGraph(G):
        InvG = [[0 for j in range(0, length(G[i]))] for i in range(0, length(G))]
        for i in range(0, length(G)):
            for j in range(0, length(G[i])):
                InvG[j][i] = G[i][j]



    def ProbTraverse(G, CurrNode, A):
        InvG = self.InvertGraph(G)
        TopoSort = self.topologicalSort(InvG, [i for i in range(0, length(InvG))])
        VTable = [0 for i in range(0, length(G))]
        VTable[R] = float(1)
        ETable = [[0 for j in range(0, length(G[i]))] for i in range(0, length(G))] #how do a make list of edges? LOL


        for i in range(0, length(TopoSort)):
            v = TopoSort[i]
            for sofincedge in range(0, length(InvG[v])):
                VTable[v] += ETable[sofincedge][v]

            for dofoutedge in range(0, length(InvG[v])):
                ETable[v][dofoutedge] = G[v][dofoutedge] * VTable[v]

        ActionProbs = [0 for action in A]
        for i in range(0, length(A)):
            if A[i] == -1:
                ActionProbs[i] = 0
            else:
                ActionProbs[i] = VProbTable[i]

        return ActionProbs

    def topologicalSortUtil(self,v,visited,stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.CGraph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)

        # Push current vertex to stack which stores result
        stack.insert(0,v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(CGraph, VList):
        # Mark all the vertices as not visited
        visited = [False]*VList
        stack =[]

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(0, Length(VList)):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)

        # Print contents of stack
        print(stack)
        return stack



    '''def select_action(self): 
        #THIS IS THE RANDOM WALK EVERY TIME ACTION IS NEEDED
        #ISSUE: need case where a truly random intervention is chosen, action may not be in table
        #ISSUE: also need to invert the graph first for convenience sake
        #go to reward node
        #propagate backwards with causal effect = probability?
        rewnode = len(g) - 1 #reward node
        actions = self.anodes #assigns action nodes
        currnode = len(g) #index of current node in graph structure
        while currnode not in anodes:
            n = self.cgraph[currnode]
            sumceffs = 0
            for ceff in n:
                sumceffs += ceff
            eachceff = []
            for ceff in n:
                eachceff.append((ceff / float(sumceffs), i))
            #by this point, each causal effect (ceff) should be normalized into a "probability"
            probsum = eachceff
            preventry = null
            for entry in probsum:
                if preventry == null:
                    probsum[0] = probsum[0]
                    preventry = probsum[0]
                else:
                    entry[0] = preventry[0] + entry[0]
            rnum = random.random()
            ind = length(probsum)
            entry = probsum[ind]
            while rnum <= entry[0]:
                ind -= 1
                entry = probsum[ind]

            currnode = entry[1]

        index = 0
        while self.anodes[index] != currnode:
            index += 1
        return index'''

    def select_action(self): 
        probsum = self.actionProbs
        preventry = 0
        for i in range(0, len(probsum)):
            probsum[i] += preventry
            preventry = probsum[i]

        rnum = random.random()
        ind = length(probsum)
        marker = probsum[ind]
        while rnum <= marker:
            ind -= 1
            marker = probsum[ind]


ENV_NAME = 'Assault-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True)
