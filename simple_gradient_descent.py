import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2 # f(x, y) = x^2 + y^2

def gradient(x, y):
    return np.array([2*x, 2*y]) # ∇f(x,y) = [2x, 2y]

def gradient_descent(start, learning_rate, iterations):
    points = [np.array(start)]
    point = np.array(start)

    for i in range(iterations):
        grad = gradient(point[0], point[1])

        point = point - learning_rate * grad

        points.append(point.copy())

    return np.array(points)

def visualize():
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = f(X, Y)

    # gradient descent from point (-4, 4)
    path = gradient_descent((-4, 4), 0.1, 20)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, 20, cmap='viridis')
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10)
    plt.plot(0, 0, 'bo', markersize=10)  # minimum

    plt.title('Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    visualize()
