import matplotlib.pyplot as plt
import numpy as np

data = np.array([[6, 7], [8, 9], [10, 13], [14, 17.5], [18, 18]])

X = data[:, :1]
y = data[:, 1:]


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

yp = model.predict(X)


score = model.score(X, y) # r-squared score

print("train score: %.6f" % score)

plt.plot(X, y, 'k.')
plt.plot(X, yp, 'r-')
plt.xlabel('size')
plt.ylabel('price')
plt.grid(True)
plt.show()