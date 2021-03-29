import matplotlib.pyplot as plt
import numpy as np

#Plotting the moving average over n games for score and m steps for loss

"""
loss = np.load("loss0.npy")
loss = np.concatenate((loss, np.load("loss.npy")))
scores = np.load("scores0.npy")
scores = np.concatenate((scores, np.load("scores.npy")))
"""
loss = np.load("loss.npy")
scores = np.load("scores.npy")
n = 10
m = 100

plt.close()
plt.plot(np.convolve(loss, np.ones(m), 'valid') / m)
plt.title("Moving average Huber loss per step during training")
plt.xlabel("# step")
plt.ylabel("loss")
plt.savefig("loss.png", dpi=600)

plt.close()
plt.plot(np.convolve(scores, np.ones(n), 'valid') / n)
plt.title("Moving average score per game during training")
plt.xlabel("# game")
plt.ylabel("score")
plt.savefig("scores.png", dpi=600)