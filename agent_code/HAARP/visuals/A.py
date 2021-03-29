import numpy as np
actions = ['UP','DOWN','LEFT','RIGHT','WAIT','BOMB']
array = np.load("step93.npy").astype(object)
for i,action in enumerate(actions):
    array[array==i] = action

print(array.T)