from __future__ import division
import math
import sys
import random

import numpy as np
import scipy.sparse as sp

import pylab
import  matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

class MDP(object):

    """A Markov Decision Process.
    Define class members
    S: [int] The number of states;
    A: [int] The number of acions;
    T: [array]
        Transition matrices. The simplest way is using a numpy 
        array that has the shape ``(A, S, S)``. Each element with index
        [a,s,s'] represent the transition probability T(s, a, s'). 
        When state or action space is overwhelmingly large and sparse,
        then ``scipy.sparse.csr_matrix`` matrices can be used.
    R: [array]
        Reward matrices or vectors. Let's use the simplest form with the 
        shape ``(S,)``. Each element with index s is the reward R(s).
        Still ``scipy.sparse.csr_matrix`` can be used instead of numpy arrays.    
    gamma: [float] Discount factor. The per time-step discount factor on future
        rewards. The value should be greater than 0 up to and including 1.
        If the discount factor is 1, then convergence cannot be assumed and a
        warning will be displayed. 
    epsilon : [float]
        Error bound. The maximum change in the value function at each
        iteration is compared against. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function.
    max_iter : [int]
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. 
    """

    def __init__(self, num_states, num_actions, transitions, rewards, discount, epsilon, max_iter):
        # Set the number of states and number of actions
        self.S = int(num_states)
        self.A = int(num_actions)
        
        # Set the maximum iteration number
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "Warning: the maximum number of iterations must be greater than 0.")
        else:
            self.max_iter = 100000
            
        # Set the discount factor
        if discount is not None:
            self.gamma = float(discount)
            assert 0.0 < self.gamma <= 1.0, (
                "Warning: discount rate must be in (0, 1]")
        else:
            self.gamma = 0.99
        # check that error bound is approperiate
        
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, (
            "Warning: epsilon must be greater than 0.")
        else:
            self.epsilon = 1E-5
            
        if transitions is not None:
            self.T = np.asarray(transitions)
            assert self.T.shape == (self.A, self.S, self.S), (
            "Warning: the shape of transition function does not match with state and action space")
        else:
            self.T = np.zeros([self.A, self.S, self.S])
            
        if rewards is not None:
            self.R = np.asarray(rewards).astype(float)
            assert self.R.shape == (self.S, ), (
                "Warning: the shape of reward function does not match with state space")
        else:
            self.R = np.random.random(self.S)
            
        # Reset the initial iteration number to zero
        self.iter = 0
        
        # Reset value matrix to None
        # Since value function is mapping from state space to real value. 
        # When it is initialized, it should be a numpy array with shape (S,)
        self.V = None
        
        # Reset Q matrix to None
        # When it is initialized, it should be a numpy array with shape (A, S)
        self.Q = None
        
        # Reset policy matrix to None
        # It should have the shape (S,). Each element is the choosen action
        self.policy = None
    def BellmanUpdate(self, V = None):
        if V is None:
            V = self.V

        try:
            assert V.shape in ((self.S,), (1, self.S)), ("Warning: shape of V is not correct")
        except AttributeError:
            raise TypeError("V must be a numpy array or matrix.")

        Q = np.empty((self.A, self.S))

        for a in range(self.A):
            Q[a] = self.R + self.gamma * self.T[a].dot(V)

        return (Q.argmax(axis = 0), Q.max(axis = 0))


class gridworld(MDP):
    # Firsly define the MDP for the gridworld. 
    # The MDP should have 8*8=64 states to represent all the states.
    # There should be 5 actions: moving left, moving up, moving right, moving down, staying.
    # Firstly initialize the transition and reward function with an all zero matrix
    def __init__(self, dimension = 8, probability = 0.8):
        super(gridworld, self).__init__(num_states = dimension**2, num_actions = 5, transitions = np.zeros([5, dimension**2, dimension**2]), 
                     rewards = np.zeros([dimension**2]), discount = 0.999, epsilon = 1e-4, max_iter = 100) 

        self.dim = dimension
        self.prob = probability
        
        self.__build_transitions__()
        self.__build_rewards__()
    
    def __coord_to_index__(self, coord):
        # Then translate the coordinate to index
        index = 0
        base = 1
        for i in range(len(coord)):
            index += coord[len(coord) - 1 - i] * base 
            base *= self.dim
        return int(index)  
    
    def __index_to_coord__(self, index):
        # Then translate the state index to coord
        return [int(index/self.dim),int(index)%int(self.dim) ]
    
    def __build_transitions__(self):
        self.T = list()
        for a in range(self.A):
            self.T.append(np.zeros([self.S, self.S]).astype(float))
            for y in range(self.dim):
                for x in range(self.dim):
                    s = self.__coord_to_index__([y, x])
                    if a == 0:
                        # Action 0 means staying
                        self.T[a][s, s] = 1.0
                        continue
                    # 20% probability of moving in random direction
                    self.T[a][s, s] += (1 - self.prob)/5.0
                    
                    # Action 4 means going up, y is reduced by 1, x doesn't change 
                    s_ = self.__coord_to_index__([abs(y-1), x])
                    self.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 4) * self.prob
                    
                    # Action 3 means going down, y doesn't change, x is reduced by 1  
                    s_ = self.__coord_to_index__([y, abs(x-1)])
                    self.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 3) * self.prob

                    # Action 2 means going down, y add 1, x doesn't change 
                    s_ = self.__coord_to_index__([self.dim - 1 - abs(self.dim - 1  - y - 1), x])
                    self.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 2) * self.prob

                    # Action 1 means going right, y does not change, x add 1
                    s_ = self.__coord_to_index__([y, self.dim - 1 - abs(self.dim - 1 - x - 1)])
                    self.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 1) * self.prob
            self.T[a][self.dim - 1] = 0.0
            self.T[a][self.dim - 1, self.dim - 1] = 1.0
            self.T[a][int(self.dim) * (int(self.dim/2) - 1) + (int(self.dim/2) - 1)] = 0.0
            self.T[a][int(self.dim) * (int(self.dim/2) - 1) + (int(self.dim/2) - 1), int(self.dim) * (int(self.dim/2) - 1) + (int(self.dim/2) - 1)] = 1.0
         
        self.T = np.asarray(self.T)
        
    def __build_rewards__(self):
        # The 64th cell with coord [7, 7] has the highest reward
        # The reward function is a radial basis function
        self.R = np.zeros([self.S])

        for s in range(self.S):
            coord = self.__index_to_coord__(s)
            self.R[s] = - 1.0 * np.linalg.norm(np.array(coord).astype(float) 
                          - np.array([self.dim - 1, self.dim - 1]).astype(float), ord = 2)
        self.R = 2.0 * np.exp(self.R).astype(float)
        
        R = np.zeros([self.S])
        for s in range(self.S):
            coord = self.__index_to_coord__(s)
            R[s] = -2.0 * np.linalg.norm(np.array(coord).astype(float) 
                          - np.array([self.dim/2 - 1, self.dim/2 - 1]).astype(float), ord = 2)
        self.R = self.R - 1.0 * np.exp(R).astype(float)
        #self.R -= (np.max(self.R) + np.min(self.R))/2.0
        self.R /= max(abs(np.max(self.R)), abs(np.min(self.R)))
        


    def draw_grids(self, rewards = None, title = None):
        # Draw the reward mapping of the grid world with grey scale
        if rewards is None:
            rewards = self.R
        R = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                R[i, j] = rewards[self.__coord_to_index__([i, j])]
        if title is None:
            title = 'Reward mapping'
        pylab.title(title)
        pylab.set_cmap('gray')
        pylab.axis([0, self.dim, self.dim, 0])
        c = pylab.pcolor(R, edgecolors='w', linewidths=1)
        pylab.show()
    
    def draw_plot(self, rewards = None, values = None, title = None):
        # Draw the reward or value plot with state indices being the x-axle
        if rewards is not None:
            plt.ylabel('Reward')
            plt.plot(range(self.S), rewards, 'r--') 
        if values is not None:
            plt.ylabel('Value')
            plt.plot(range(self.S), values, 'b--')
        plt.xlabel('State Index')
        plt.show()
        
   
    def draw_policy(self, policy = None, save = False):
        # Draw the policy mapping of the grid world
        if policy is None:
            policy = self.policy
        if save:
            plt.switch_backend('agg')

        fig, ax = plt.subplots()
        plt.axis([0, self.dim, self.dim, 0])
        

        grey = ax.get_facecolor()
        colors = ['black', 'red', 'yellow', 'green', 'blue', grey]
        actions = ['stay', 'right', 'down', 'left', 'up', 'unknown']
        for a in range(len(colors)):
            x = list()
            y = list()
            states = (policy==a).nonzero()[0]
            
            for s in states:
                [y_, x_] = self.__index_to_coord__(s)
                y.append(y_ + 0.5)
                x.append(x_ + 0.5)
            if actions[a] == 'unknown':
                edgecolor = 'black'
            else:
                edgecolor = 'none'
            ax.scatter(x, y, c=colors[a], label=actions[a],
                       alpha=0.8, edgecolors= edgecolor)

        ax.legend()
        #ax.grid(True)
        minor_ticks = np.arange(0, self.dim, 1)
        ax.set_xticks(minor_ticks, minor = True)
        ax.set_yticks(minor_ticks, minor = True)
        ax.grid(which='minor', color = 'white', linestyle = '--')

        if save:
            plt.savefig(save)
        else:
            plt.show()

class PolicyIteration():
    
    ##Design a Policy Iteration algorithm for a given MDP
    
    def __init__(self, MDP, policy_init = None):
        ## Reset the current policy
        
        self.M = MDP
        self.iter = 0
        
        # Check if the user has supplied an initial policy.
        if policy_init is None:
            # Initialise a policy that greedily maximises the one-step reward
            self.M.policy, _ = self.M.BellmanUpdate(np.zeros(self.M.S))
        else:
            # Use the provided initial policy
            self.M.policy = np.array(policy_init)
            # Check the shape of the provided policy
            assert self.policy.shape in ((self.M.S, ), (self.M.S, 1), (1, self.M.S)), \
                ("Warning: initial policy must be a vector with length S.")
            # reshape the policy to be a vector
            self.M.policy = self.M.policy.reshape(self.M.S)
            # The policy must choose from the action space
            msg = "Warning: action out of range."
            assert not np.mod(self.M.policy, 1).any(), msg
            assert (self.M.policy >= 0).all(), msg
            assert (self.M.policy < self.M.A).all(), msg
        # set the initial values to zero
        self.M.V = np.zeros(self.M.S)
    
    
    def TransitionUpdate(self, policy = None):
        # Compute the transition matrix under the current policy.
        #
        # The transition function MDP.T is a (A, S, S) tensor,
        # The actions in the first dimension are undeterministic.
        #
        # Now the action is determined by the policy
        # The transition function becomes a (S,S) matrix, named P
        #
        # Use the current policy to find out P
        if policy is None:
            policy = self.M.policy
        P = np.empty((self.M.S, self.M.S))
        for a in range(self.M.A):
            indices = (policy == a).nonzero()[0]
            if indices.size > 0:
                P[indices, :] = self.M.T[a][indices, :]
        return P
    
    def ValueUpdate(self, epsilon = 1E-10, max_iter = 10000):
        if epsilon is None:
            epsilon = self.M.epsilon
        if max_iter is None:
            maxtier = self.M.max_iter

        # The transition probability is determined by the policy
        P = self.TransitionUpdate()
        assert P.shape == (self.M.S, self.M.S)

        #Reset the Value function to be equal to the Reward function
        self.M.V = self.M.R.copy()

        itr = 0

        while True:
            itr +=1

            V_temp = self.M.V.copy()
            self.M.V = self.M.R + self.M.gamma * P.dot(self.M.V)
            err = np.absolute(self.M.V - V_temp).max()
            if err < epsilon or itr >= max_iter:
                break

        return self.M.V

    def ValueUpdate_LQ(self):
        # Calculate the value function of the policy by solving a linear equation.
        #
        # Observe the Bellman Equation. 
        # The policy, rewards, transition probabilities are all given. 
        # Can you solve the value function(matrix) by solving a linear equation?
        # Think about how to do it. 
        
        P = self.TransitionUpdate()
        assert P.shape == (self.M.S, self.M.S)

        self.M.V = np.linalg.solve((sp.eye(self.M.S, self.M.S) - self.M.gamma * P), self.M.R)
        return self.M.V

    def iterate(self, LQ = False):
        # Run the policy iteration algorithm.
        V_ = np.zeros([self.M.S])
        while True:
            self.iter += 1
            # Calculate the value function resulted from the curretn policy
            # attribute
            if LQ:
                self.ValueUpdate_LQ()
            else:
                self.ValueUpdate()
            
            # Make one step improvement on the policy based on current value function.
            policy_, _ = self.M.BellmanUpdate()
            #print(policy_)
            #print(self.V)
            #print(V_)
            # calculate the difference between the newly generated policy and the current policy
            err = (policy_ != self.M.policy).sum()
            #err = np.absolute(self.V - V_).max()
            
            # If the difference is smaller than the error bound MDP.epsilon, then stop;
            # Otherwise if the maximum number of iterations has been reached, then stop;
            # Otherwise update the current policy with the newly generated policy
            if err <= self.M.epsilon:
                break
            elif self.iter == self.M.max_iter:
                break
            else:
                self.M.policy = policy_
                V_ = self.M.V.copy()

class wrapper(object):
    def __init__(self, game):
        self.s = 0

        self.game = game

        self.observation_space = np.asarray([1])
        self.action_space = np.asarray([1])

    def seed(self, seed):
        return random.seed(seed)

    @property
    def num_actions(self):
        return self.game.A

    @property
    def num_states(self):
        return self.game.S

    @property
    def reward_range(self):
        return np.max(np.abs(self.game.R))
    
    def reset(self):
        self.s = 0
        return np.array([self.s])

    def step(self, a):
        if not isinstance(a, np.ndarray):
            a = np.asarray([a])
        a = np.round(np.clip(a, 0, self.num_actions - 1)).astype(int)[0] 
        assert 0.0 <= a < self.num_actions, ("Warning: action %d not in range" % a)

        p = np.reshape(self.game.T[a, self.s], [self.game.S])
        s_ = np.random.choice(self.game.S, 1, p = p)
        if isinstance(s_, np.ndarray):
            s_ = s_.flatten()[0]
        self.s = int(s_)

        done = False
        if (self.s == self.game.S - 1) or (self.s == int(self.game.dim) * (int(self.game.dim/2) - 1) + int(self.game.dim/2) - 1):
            done = True

        return np.array([self.s]).astype(float), self.game.R[int(self.s)], done, None

    def render(self):
        self.render_policy()

    def render_rewards(self):
        self.game.draw_grids()

    def render_policy(self, policy = None):
        if policy is None:
            policy = self.game.policy
        self.game.draw_policy(np.asarray(policy))
    
    def close(self):
        self.game = None

