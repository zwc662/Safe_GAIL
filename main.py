import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

from mdp import MDP, gridworld, PolicyIteration, wrapper 
import logging
import time
import random
import os

import pickle


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='batch_size (default: 15000')
parser.add_argument('--buffer-size', type=int, default=150000, metavar='N',
                    help='buffer_size (default: 15000')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--load-policy', default = False, metavar='G',
                    help='load policy net checkpoint file path')
parser.add_argument('--load-value', default = False, metavar='G',
                    help='load value net checkpoint file path')
parser.add_argument('--train', action = 'store_true',
                    help='Reload!!!!')
parser.add_argument('--test', action = 'store_true',
                    help='Fire!!!!')
parser.add_argument('--demo', action = 'store_true',
                    help='Show me!!!')

args = parser.parse_args()

assert (args.test is False) or (args.train is False) or (args.demo is False), \
        'Please choose at least one from train and test and demo?'

assert not((args.test is True) and (args.train is True)), 'Both test and train are chosen. Please choose one' 

assert not((args.demo is True) and (args.train is True)), 'Both demo and train are chosen. Please choose one' 

discrete = False
if discrete:
    logging.info("Using discrete actions ---> multiple outputs")

if args.env_name == "gridworld":
    env = wrapper(gridworld(8))
elif os.path.isfile(args.env_name): 
    env = pickle.load(args.load_env)
else:
    env = gym.make(args.env_name)


num_inputs = env.observation_space.shape[0]
if discrete:
    num_actions = env.num_actions
else:
    num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions, discrete)
value_net = Value(num_inputs)

if args.load_policy:
    policy_net.load_state_dict(torch.load(args.load_policy)) 

if args.load_value:
    value_net.load_state_dict(torch.load(args.load_value))

def select_action(state, discrete = False):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))

    if discrete:
        action = torch.argmax(action_mean, dim = 1).reshape([1, 1])
        return action, action, action_std
    else:
        action = torch.normal(action_mean, action_std)
    return action, action_mean, action_std

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)



filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
pathname = os.path.join(os.path.dirname(__file__), "tmp/" + filename)
os.makedirs(pathname)

logging.basicConfig(level = logging.DEBUG, \
        format = '%(asctime)s %(levelname)s %(message)s', \
        filename = os.path.join(pathname, filename + '.log'), \
        filemode = 'w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



if args.env_name == 'gridworld':
    running_state = ZFilter((num_inputs,), demean = False, destd = False, clip= 2.0 * env.num_states)
    running_reward = ZFilter((1,), demean=False, destd = False, clip = 2.0 * env.reward_range)
    policy = np.empty([env.num_states])
    policy[-1] = 0

    file_grid = open(os.path.join(pathname, 'gridworld.pkl'), 'bw')
    pickle.dump(env, file_grid)
else:
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

for i_episode in count(1):
    memory = Memory(args.buffer_size)

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        # batch_size is the total number of t
        # num_steps is incremented by max(t)=10000 in every iteration
        state_ = env.reset()
        state = running_state(state_)

        reward_sum = 0
        max_step = 10000
        if args.env_name == 'gridworld':
            max_step = env.num_states * env.num_states 
        for t in range(max_step): # Don't infinite loop while learning
            #if i_episode % 10 == 0:
            #    action = torch.randint(0, env.num_actions, [1, 1]).data[0].numpy() 
            #else: 
            action, action_mean, _ = select_action(state, discrete)
            action = action.data[0].numpy()

            if args.env_name == 'gridworld':
                action_mean = action_mean.data[0].numpy()
                policy[int(state_[0])] = np.clip(action_mean, 0, env.num_actions - 1).round().astype(int)[0]

                if reward_sum <= 0.0:
                    if random.randint(0, 2) <= 1.0:
                        action = torch.argmax(torch.randn([1, 4]), dim = 1).reshape([1, 1]).data[0].numpy()

            next_state_, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state_)

            mask = 1
            if done:
                mask = 0
                if args.env_name == 'gridworld':
                    policy[int(next_state_[0])] = 0

            if not args.test:
                demo = False
                if args.demo:
                    demo = True
                memory.push(state, np.array([action]), mask, next_state, reward, demo = demo)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
            state_ = next_state_

        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes

    if args.train:
        batch = memory.sample()
        update_params(batch)

    if i_episode % args.log_interval == 0:
        logging.info('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
    if i_episode % 100 == 1:
        if args.train:
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
            torch.save(value_net.state_dict(), os.path.join(pathname, filename + '_episode_{}_value.pth'.format(i_episode)))
            torch.save(policy_net.state_dict(), os.path.join(pathname, filename + '_episode_{}_policy.pth'.format(i_episode)))
        if args.env_name == 'gridworld':
            env.game.draw_policy(policy = policy, save = os.path.join(pathname, filename + '_episode_{}.png'.format(i_episode)))
