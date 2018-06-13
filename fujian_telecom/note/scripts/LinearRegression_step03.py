import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# data = np.array([[6, 7], [8, 9], [10, 13], [14, 17.5], [18, 18]])
data = np.array([[6, 4, 7], [8, 3.5, 9], [10, 6, 13], [14, 5, 17.5], [18, 3.0, 18]])

X = data[:, :2]
y = data[:, 2:]


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

yp = model.predict(X)


score = model.score(X, y) # r-squared score

print("train score: %.6f" % score)


x0 = np.linspace(6, 18, 10)
x1 = np.linspace(3, 6, 10)
xx0, xx1 = np.meshgrid(x0, x1)
xx = zip(xx0.flatten(), xx1.flatten())

yp = model.predict(xx)
zz = yp.reshape((10, -1))


ax = plt.gca(projection='3d')
ax.plot(X[:,0], X[:,1], y[:,0], 'k.')
ax.plot_surface(xx0, xx1, zz)

ax.set_yticks([3, 4, 5])
ax.set_xlabel('size')
ax.set_ylabel('rank')
ax.set_zlabel('price')
plt.grid(True)
plt.show()