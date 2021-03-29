from collections import namedtuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import events as e
import settings as s
from .callbacks import gen_state
from .callbacks import DQN

###############
# Deep Q Learning (DQN) Algorithm
###############

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    Class storing previous Transitions between an old and a new state with
    the corresponding action and reward.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, BATCH_SIZE):
        """
        Returns a random sample with size BATCH_SIZE
        """
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)
        
def optimize_model(self):
    """
    Function to run one optimization step
    """
    if len(self.memory) < self.BATCH_SIZE:
        return
    transitions = self.memory.sample(self.BATCH_SIZE)
    #Transpose the batch
    batch = Transition(*zip(*transitions))

    #Get the non-final transitions (important for later)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Compute Q(s_t, a)
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    #Compute V(s_{t+1}) for all next states (Only for non-final transitions; for final ones it's 0)
    next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    #Compute the expected Q values
    expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
    
    #Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    self.loss.append(loss.item())

    #Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    #Gradient clipping
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

def gen_reward(self, events, customs=[]):
    """
    Reward function to en/discourage certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT : 3,
        e.COIN_COLLECTED : 2,
        e.INVALID_ACTION : -0.1,
        e.GOT_KILLED : -3,
        e.KILLED_SELF : -3
    }
    reward = -0.1 #general penalty for taking actions
    for event in events:
        if event in game_rewards:
            reward += game_rewards[event]
    for r in customs:
        reward += r
    return torch.tensor([reward], device=self.device)

###############
# Interface
###############

def setup_training(self):
    """
    Initialise self for training purpose.
    """
    self.rev_ACTIONS = {v:k for k,v in self.ACTIONS.items()}

    #Define training hyperparameters
    self.BATCH_SIZE = 128
    self.GAMMA = 0.95
    self.EPS_START = 1
    self.EPS_END = 0.05
    self.EPS_DECAY = 1000
    self.eps_threshold = self.EPS_START
    self.TARGET_UPDATE = 10
    self.SAVE_NN = 100

    self.scores = []
    self.loss = []

    self.optimizer = optim.Adam(self.policy_net.parameters())
    #Load old checkpoint to resume training
    try:
        self.optimizer.load_state_dict(torch.load("HAARP.tar")["optimizer"])
        self.old_rounds = torch.load("HAARP.tar")["rounds"]
    except:
        self.old_rounds = 0
    self.memory = ReplayMemory(10000)
    self.target_net = DQN(len(self.ACTIONS)).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    #If first round, return
    if self_action is None:
        return
    #Custom rewards
    c_rewards = []

    old_state = gen_state(self, old_game_state)
    if e.BOMB_DROPPED in events:
        #crate reward when bomb placed
        new_state, bombed_crates = gen_state(self, new_game_state, True)
        c_rewards.append(0.1 * len(bombed_crates))
    else:
        new_state = gen_state(self, new_game_state)
    
    action = torch.tensor([self.rev_ACTIONS[self_action]], device=self.device).unsqueeze(0)
    # Store the transition in memory
    self.memory.push(old_state, action, new_state, gen_reward(self, events, c_rewards))
    # Perform one step of the optimization
    optimize_model(self)
        
def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    #Store final transition of the round
    if last_action is not None:
        action = torch.tensor([self.rev_ACTIONS[last_action]], device=self.device).unsqueeze(0)
        # Store the transition in memory
        self.memory.push(gen_state(self, last_game_state), action, None, gen_reward(self, events))

        #store score
        self.scores.append(last_game_state["self"][1])

    # Perform one step of the optimization
    optimize_model(self)

    self.eps_threshold = self.EPS_END + self.EPS_END * math.cos((last_game_state["round"] + self.old_rounds) / math.pi / 78) + \
            (self.EPS_START - 2 * self.EPS_END) * \
            math.exp(-1. * (last_game_state["round"] + self.old_rounds) / self.EPS_DECAY)

    # Update the target network, copying all weights and biases in DQN
    if last_game_state["round"] % self.TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if len(self.scores) and len(self.loss):
            print()
            print("SCORE:", "mean:", np.mean(self.scores), "min:", min(self.scores), "max:", max(self.scores))
            print("LOSS:", "mean:", np.mean(self.loss), "min:", min(self.loss), "max:", max(self.loss))
        np.save("loss", np.array(self.loss))
        np.save("scores", np.array(self.scores))

    #Save the target network
    if last_game_state["round"] % self.SAVE_NN == 0:
        torch.save({"model":self.target_net.state_dict(), 
                    "optimizer":self.optimizer.state_dict(), 
                    "rounds":last_game_state["round"]}, "HAARP.tar")