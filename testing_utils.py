import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_test_mesh(positions, face, block=True):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(positions.shape[0]):
	    ax.scatter(positions[i,0], positions[i,1], positions[i,2], label=str(i))
	    ax.text(positions[i,0], positions[i,1], positions[i,2], '%d' % int(i))

	for f in face:
	    f_ = np.hstack([f, f[0]])
	    ax.plot(positions[f_, 0], positions[f_, 1], positions[f_, 2])

	plt.show(block=block)