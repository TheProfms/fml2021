import random
import math

import numpy as np
import torch
import torch.nn as nn

import settings as s

###############
# Deep Q Learning (DQN) Algorithm
###############

class DQN(nn.Module):
    """
    Class to define the neural network
    """
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.maxpool2 = nn.MaxPool2d(2)

        self.convNF = nn.Conv2d(5, 32, 3)

        self.l1 = nn.Linear(128 * 5 * 5 + 32 * 3 * 3, 1024)
        self.out = nn.Linear(1024, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        y = x[:,:,12:17,12:17]
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        y = self.convNF(y)

        x = torch.cat((x.flatten(1), y.flatten(1)), 1)
        
        elu = nn.ELU()
        x = elu(self.l1(x.flatten(1)))
        return self.out(x.view(x.size(0), -1))

def get_explosion_indices(field, bomb_x, bomb_y, radius):
    """
    Finds the coordinates of the field reached by the bombs explosion
    """
    indices = []
    radius += 1
    for x in range(bomb_x, bomb_x + radius):
        if field[x][bomb_y] == -1:
            break
        indices.append((x, bomb_y))
    for x in range(bomb_x, bomb_x - radius, -1):
        if field[x][bomb_y] == -1:
            break
        indices.append((x, bomb_y))
    for y in range(bomb_y, bomb_y + radius):
        if field[bomb_x][y] == -1:
            break
        indices.append((bomb_x, y))
    for y in range(bomb_y, bomb_y - radius, -1):
        if field[bomb_x][y] == -1:
            break
        indices.append((bomb_x, y))
    return indices

def transform_coords(self_pos, pos):
    """
    Transforms the coordinates of the 17x17 grid to our 29x29 grid
    """
    return (pos[0] - self_pos[0] + 14, pos[1] - self_pos[1] + 14)

def gen_state(self, game_state, ret_bombed_crates=False):
    """
    Generates the state vector for the neural network based on game_state.

    Used: field, bombs, explosion_map, coins, position of self and others
    Unused: round, step, current score and bomb action of self and others
    """
    #Define empty state vector
    state = torch.zeros((5,29,29), dtype=torch.float, device=self.device)
    if game_state is None:
        return state.unsqueeze(0)

    self_pos = game_state["self"][3]
    start = transform_coords(self_pos, (1,1))
    end = transform_coords(self_pos, (16,16))

    bombed_crates = []
    
    #One 29x29 grid for position of walls
    state[0][start[0]:end[0], start[1]:end[1]] = -1 * torch.tensor(game_state["field"][1:16, 1:16] != -1, device=self.device)

    #One 29x29 grid for position of crates
    state[1][start[0]:end[0], start[1]:end[1]] = torch.tensor(game_state["field"][1:16, 1:16] == 1, device=self.device)
    
    #One 29x29 grid for position of coins
    for coin in game_state["coins"]:
        state[2][transform_coords(self_pos, coin)] = 1
    
    #One 29x29 grid for position of other agents
    for other in game_state["others"]:
        state[3][transform_coords(self_pos, other[3])] = 1
    
    #One 29x29 grid for bomb positions and their splashzone
    state[4][start[0]:end[0], start[1]:end[1]] = torch.tensor(game_state["explosion_map"][1:16, 1:16], device=self.device)
    for bomb in game_state["bombs"]:
        threat_level = (s.BOMB_TIMER - bomb[1]) / s.BOMB_TIMER
        for threat_field in get_explosion_indices(game_state["field"], bomb[0][0], bomb[0][1], s.BOMB_POWER):
            threat_field = transform_coords(self_pos, threat_field)
            state[4][threat_field] = max(threat_level, state[4][threat_field])
            if bomb[0] == self_pos and state[1][threat_field] == 1:
                bombed_crates.append(threat_field)
    if ret_bombed_crates:
        return state.unsqueeze(0), bombed_crates
    return state.unsqueeze(0)

###############
# Interface
###############

def setup(self):
    """
    Called once before a set of games to initialize data structures etc.
    """
    #Sets device to either CPU or GPU (CUDA)
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    self.ACTIONS = {i:k for i,k in enumerate(s.INPUT_MAP.values())}

    self.last_actions = ['WAIT','WAIT','WAIT','BOMB']

    #Setup of the network. Tries to load parameters from file
    self.policy_net = DQN(len(self.ACTIONS))
    try:
        self.policy_net.load_state_dict(torch.load("HAARP.tar")["model"])
    except:
        pass
    self.policy_net.to(self.device)
    if self.train:
        self.policy_net.train()
    else:
        self.policy_net.eval()

    #initial calls of the network to allocate memory
    self.policy_net(gen_state(self, None))
    self.policy_net(gen_state(self, None))

def act(self, game_state):
    """
    Called each game step to determine the agent's next action.
    """
    if self.train and random.random() < self.eps_threshold:
        #random action
        return np.random.choice(list(self.ACTIONS.values()), p=[.2, .2, .2, .2, .05, .15])
    else:
        with torch.no_grad():
            #Returning the action with highest predicted reward
            pred = self.policy_net(gen_state(self, game_state))
            action = self.ACTIONS[pred.max(1)[1].view(1, 1).item()]
            if all([a == self.last_actions[0] for a in self.last_actions]):
                action = 'BOMB'
            self.last_actions.pop(0)
            self.last_actions.append(action)

            #record action for all possible positions of self
            """
            predictions = np.full((17,17), -1)
            for pos in np.transpose(np.nonzero(game_state["field"]==0)):
                info = game_state["self"]
                game_state["self"] = (info[0], info[1], info[2], tuple(pos))
                predictions[pos[0], pos[1]] = self.policy_net(gen_state(self, game_state)).max(1)[1].view(1, 1).item()
            np.save("actions/step{}".format(game_state["step"]), predictions)
            """
            return action
