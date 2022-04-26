# Imports and Packages
import math

import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols, diff

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""
# Function


def f(x):
    return x**2 + x + 1

# Slope and Derivatives


def df(x):
    return (2 * x) + 1

# Make Data


x_1 = np.linspace(start=-3, stop=3, num=500)

# Plot
plt.figure(figsize=[10, 5])

plt.subplot(1, 2, 1)

plt.title("Cost Function")
plt.xlim(-3, 3)
plt.ylim(0, 8)
plt.xlabel("x", fontsize=16)
plt.ylabel("f(x)", fontsize=16)
plt.plot(x_1, f(x=x_1))

plt.subplot(1, 2, 2)

plt.title("Derivative of Cost Function")
plt.xlim(-2, 3)
plt.ylim(-3, 6)
plt.xlabel("x", fontsize=16)
plt.ylabel("f'(x)", fontsize=16)
plt.plot(x_1, df(x=x_1))

plt.grid()
plt.show()

# Python Loops and Gradient Descent
new_x = -3
previous_x = 0
step_multiplier = 0.1
precision = 0.0001
frequency = 0

x_list = [new_x]
slope_list = [df(new_x)]

while True:
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - (step_multiplier * gradient)

    # print(f'previous_x = {previous_x}, \t\t\t new_x = {new_x}, \t\t\t gradient = {gradient}
    # \t\t\t diff = {(step_multiplier * gradient)}')

    x_list.append(new_x)
    slope_list.append(df(new_x))

    step_size = abs(new_x - previous_x)
    frequency += 1
    if step_size < precision:
        break

plt.figure(figsize=[10, 5])

plt.subplot(1, 3, 1)

plt.title("Cost Function")
plt.xlim(-3, 3)
plt.ylim(0, 8)
plt.xlabel("x", fontsize=16)
plt.ylabel("f(x)", fontsize=16)
plt.plot(x_1, f(x=x_1), alpha=0.8)


values = np.array(x_list)
plt.scatter(x_list, f(values), color='red', alpha=0.6)

plt.subplot(1, 3, 2)

plt.title("Derivative of Cost Function")
plt.xlim(-2, 3)
plt.ylim(-3, 6)
plt.xlabel("x", fontsize=16)
plt.ylabel("f'(x)", fontsize=16)
plt.plot(x_1, df(x=x_1), alpha=0.6)

plt.scatter(x_list, slope_list, color='red', alpha=0.6)

plt.grid()

plt.subplot(1, 3, 3)

plt.title("Gradient Descent")
plt.xlim(-0.55, -0.2)
plt.ylim(-0.15, 0.1)
plt.xlabel("x", fontsize=16)
plt.ylabel("f'(x)", fontsize=16)
plt.plot(x_1, df(x=x_1), alpha=0.6)

plt.scatter(x_list, slope_list, color='red', alpha=0.6)
plt.grid()

plt.show()

print(f"\n\nLocal Min at: {new_x} \nSlope is {df(new_x)} \nCost is {f(new_x)}\nFrequency {frequency}")


# Example 2

x_2 = np.linspace(-2, 2, 1000)


def g(x):
    return x**4 - 4 * x**2 + 5


def dg(x):
    return 4 * x**3 - 8 * x


# Plot
plt.figure(figsize=[10, 5])

plt.subplot(1, 2, 1)

plt.title("Cost Function")
plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)
plt.xlabel("x", fontsize=16)
plt.ylabel("g(x)", fontsize=16)
plt.plot(x_2, g(x=x_2))

plt.subplot(1, 2, 2)

plt.title("Derivative of Cost Function")
plt.xlim(-2, 2)
plt.ylim(-6, 8)
plt.xlabel("x", fontsize=16)
plt.ylabel("g'(x)", fontsize=16)
plt.plot(x_2, dg(x=x_2))
plt.grid()

plt.show()


def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.0001):
    new_x = initial_guess

    frequency = 0

    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    while True:
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - (multiplier * gradient)

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        step_size = abs(new_x - previous_x)
        frequency += 1
        if step_size < precision:
            return new_x, x_list, slope_list


local_min, list_x, derivative_list = gradient_descent(
    derivative_func=dg,
    initial_guess=1.8
)

print(f"\n\nLocal Min at: {local_min} \nSlope is {derivative_list[-1]}\nFrequency {len(list_x)}")


# Plot
plt.figure(figsize=[10, 5])

plt.subplot(1, 2, 1)

plt.title("Cost Function")
plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)
plt.xlabel("x", fontsize=16)
plt.ylabel("g(x)", fontsize=16)
plt.plot(x_2, g(x=x_2))

values = np.array(list_x)
plt.scatter(list_x, g(values), color='red', alpha=0.6)


plt.subplot(1, 2, 2)

plt.title("Derivative of Cost Function")
plt.xlim(-2, 2)
plt.ylim(-6, 8)
plt.xlabel("x", fontsize=16)
plt.ylabel("g'(x)", fontsize=16)
plt.plot(x_2, dg(x=x_2))
plt.grid()
plt.scatter(list_x, derivative_list, color='red', alpha=0.6)

plt.show()



# Example 3 - Divergence, Overflow and Python Tuples


x_3 = np.linspace(-2.5, 2.5, 1000)


def h(x):
    return x ** 4 - 4 * x ** 2 + 5
    

def dh(x):
    return 5 * x**4 - 8 * x**3


def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.0001, max_iter=300):
    new_x = initial_guess

    frequency = 0

    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    while True:
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - (multiplier * gradient)

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        step_size = abs(new_x - previous_x)
        frequency += 1
        if step_size < precision or frequency >= max_iter:
            return new_x, x_list, slope_list


local_min, list_x, derivative_list = gradient_descent(
    derivative_func=dh,
    initial_guess=-0.2,
    max_iter=71
)

print(f"\nLocal Min at: {local_min} \nSlope is {derivative_list[-1]}\nCost is {h(local_min)}\nFrequency {len(list_x)}")


# Plot
plt.figure(figsize=[10, 5])

plt.subplot(1, 2, 1)

plt.title("Cost Function")
plt.xlim(-1.2, 2.5)
plt.ylim(-1, 4)
plt.xlabel("x", fontsize=16)
plt.ylabel("h(x)", fontsize=16)
plt.plot(x_3, h(x=x_3))

values = np.array(list_x)
plt.scatter(list_x, h(values), color='red', alpha=0.6)


plt.subplot(1, 2, 2)

plt.title("Derivative of Cost Function")
plt.xlim(-1, 2)
plt.ylim(-4, 5)
plt.xlabel("x", fontsize=16)
plt.ylabel("h'(x)", fontsize=16)
plt.plot(x_3, dh(x=x_3))
plt.grid()
plt.scatter(list_x, derivative_list, color='red', alpha=0.6)

plt.show()

# help(sys)
print(sys.version)
print(sys.float_info.max)


# Tuple
breakfast = 'bacon', 'eggs', 'avocado'
unlucky_numbers = 13, 4, 26, 17

print('I love', breakfast[0])
print(f'My hotel has no {unlucky_numbers[1]}th floor')



# The Learning Rate

x_2 = np.linspace(-2, 2, 1000)


def g(x):
    return x**4 - 4 * x**2 + 5


def dg(x):
    return 4 * x**3 - 8 * x


def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.0001, max_iter=300):
    new_x = initial_guess

    frequency = 0

    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    while True:
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - (multiplier * gradient)

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        step_size = abs(new_x - previous_x)
        frequency += 1
        if step_size < precision or frequency >= max_iter:
            return new_x, x_list, slope_list


n=100

low_gamma = gradient_descent(
    derivative_func=dg,
    initial_guess=3,
    multiplier=0.0005,
    precision=0.0001,
    max_iter=n
)

mid_gamma = gradient_descent(
    derivative_func=dg,
    initial_guess=3,
    multiplier=0.001,
    precision=0.0001,
    max_iter=n
)

high_gamma = gradient_descent(
    derivative_func=dg,
    initial_guess=3,
    multiplier=0.002,
    precision=0.0001,
    max_iter=n
)

insane_gamma = gradient_descent(
    derivative_func=dg,
    initial_guess=1.9,
    multiplier=0.25,
    precision=0.0001,
    max_iter=n
)

# Plot
plt.figure(figsize=[20, 10])

plt.title("Effect of Learning Rate", fontsize=17)
plt.xlim(0, n)
plt.ylim(0, 50)
plt.xlabel("Number of Iterations", fontsize=16)
plt.ylabel("Cost", fontsize=16)

low_values = np.array(low_gamma[1])
mid_values = np.array(mid_gamma[1])
high_values = np.array(high_gamma[1])
insane_values = np.array(insane_gamma[1])
iteration_list = list(range(0, n+1))

plt.plot(iteration_list, g(low_values), color="lightgreen", linewidth=5)

plt.plot(iteration_list, g(mid_values), color="steelblue", linewidth=5)

plt.plot(iteration_list, g(high_values), color="hotpink", linewidth=5)

plt.plot(iteration_list, g(insane_values), color="red", linewidth=5)


plt.scatter(iteration_list, g(low_values), color='lightgreen', s=100)

plt.scatter(iteration_list, g(mid_values), color='steelblue', s=100)

plt.scatter(iteration_list, g(high_values), color='hotpink', s=100)

plt.scatter(iteration_list, g(insane_values), color='red', s=100)


plt.show()


# Example 4 - 3D Charts


def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1/(r + 1)


x_4 = np.linspace(start=-2, stop=2, num=200)

y_4 = np.linspace(start=-2, stop=2, num=200)

x_4, y_4 = np.meshgrid(x_4, y_4)

# Plot

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x, y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap=cm.Spectral, alpha=0.8)

plt.show()

# Partial Derivatives and Symbolic Computation

a, b = symbols('x, y')

print(f"Cost function: {f(a, b)}")
print(f"Partial derivative with respect to x: {diff(f(a, b), a)}")

print(f"Cost at x=1.8, y=1.0: {f(a, b).evalf(subs={a: 1.8, b: 1.0})}")
print(f"Partial Derivative at at x=1.8, y=1.0: {diff(f(a, b), a).evalf(subs={a: 1.8, b: 1.0})}")

# Batch Gradient Descent with SymPy

multiplier = 0.1
max_iter = 200
params = np.array([1.8, 1.0])

for n in range(max_iter):
    gradient_x = diff(f(a, b), a).evalf(subs={a: params[0], b: params[1]})
    gradient_y = diff(f(a, b), b).evalf(subs={a: params[0], b: params[1]})
    gradients = np.array([gradient_x, gradient_y])
    params = params - (multiplier * gradients)

print(f"Values in gradient array {gradients}, Minimum at x: {params[0]}, Minimum at y: {params[1]}")
print(f"Cost: {f(params[0], params[1])}")




def fpx(x, y):
    return (2*x*math.log(3)*3**(-x**2 - y**2))/((3**(-x**2 - y**2) + 1)**2)


def fpy(x, y):
    return (2*y*math.log(3)*3**(-x**2 - y**2))/((3**(-x**2 - y**2) + 1)**2)


print(f"Cost at x=1.8, y=1.0: {f(1.8, 1.0)}")
print(f"Partial Derivative at at x=1.8, y=1.0: {fpx(1.8, 1.0)}")

# Batch Gradient Descent with SymPy

multiplier = 0.1
max_iter = 500
params = np.array([1.8, 1.0])

for n in range(max_iter):
    gradient_x = fpx(x=params[0], y=params[1])
    gradient_y = fpy(x=params[0], y=params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - (multiplier * gradients)

print(f"Values in gradient array {gradients}, Minimum at x: {params[0]}, Minimum at y: {params[1]}")
print(f"Cost: {f(params[0], params[1])}")



# Graphing 3D Gradient Descent and Advanced Numpy Arrays


print(f"Cost at x=1.8, y=1.0: {f(1.8, 1.0)}")
print(f"Partial Derivative at at x=1.8, y=1.0: {fpx(1.8, 1.0)}")

multiplier = 0.1
max_iter = 500
params = np.array([1.8, 1.0])
values_array = params.reshape(1, 2)

for n in range(max_iter):
    gradient_x = fpx(x=params[0], y=params[1])
    gradient_y = fpy(x=params[0], y=params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - (multiplier * gradients)
    # values_array = np.append(arr=values_array, values=params.reshape(1, 2), axis=0)
    values_array = np.concatenate((values_array, params.reshape(1, 2)), axis=0)

print(f"Values in gradient array {gradients}, Minimum at x: {params[0]}, Minimum at y: {params[1]}")
print(f"Cost: {f(params[0], params[1])}")

# Plot

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x, y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap=cm.Spectral, alpha=0.4)
ax.scatter(values_array[:, 0], values_array[:, 1], f(x=values_array[:, 0], y=values_array[:, 1]), s=50, color='green')

plt.show()

# Advanced Numpy Array

kirk = np.array([['Captain', 'Guitar']])
print(kirk.shape)

hs_band = np.array([['Black Thought', 'MC'], ['Questlove', 'Drums']])
print(hs_band.shape)

print(f"hs_band[0] {hs_band[0]}")
print(f"hs_band[0][1] {hs_band[0][1]}")
print(f"hs_band[1][0] {hs_band[1][0]}")

the_roots = np.append(arr=hs_band, values=kirk, axis=0)
print(the_roots)

the_roots = np.append(arr=the_roots, values=[['Malik B', 'MC']], axis=0)

print(the_roots[:, 1])

"""

# Example 5 - MSE

x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5 = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7, 1)

regr = LinearRegression()

regr.fit(x_5, y_5)

print(f"intercept: {regr.intercept_[0]}, gradient: {regr.coef_[0][0]}")

# Plot

plt.scatter(x_5, y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color='red', linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

# MSE

y_hat = regr.intercept_[0] + regr.coef_[0][0]*x_5


def mse1(y, y_hat):
    return 1/y.size * sum((y - y_hat)**2)


def mse2(y, y_hat):
    return np.average((y - y_hat)**2, axis=0)


def mse3(y, y_hat):
    return mean_squared_error(y, y_hat)


print(mse1(y=y_5, y_hat=y_hat))
print(mse2(y_5, regr.predict(x_5)))

# 3d Plot MSE

nr_thetas = 500

th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)

plot_cost = np.zeros((nr_thetas, nr_thetas))

for row in range(nr_thetas):
    for col in range(nr_thetas):
        # print(plot_t0[row][col])
        y_hat = plot_t0[row][col] + plot_t1[row][col] * x_5
        plot_cost[row][col] = mse1(y_5, y_hat)

print(f"Shape t0 {plot_t0.shape}")
print(f"Shape t1 {plot_t1.shape}")
print(f"Shape plot_cost {plot_cost.shape}")
#
# fig = plt.figure(figsize=[20, 20])
# ax = fig.add_subplot(projection='3d')
#
# ax.set_xlabel('Theta 0', fontsize=20)
# ax.set_ylabel('Theta 1', fontsize=20)
# ax.set_zlabel('Cost - MSE', fontsize=20)
#
# ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.Spectral)
# plt.show()

print(f'Min value of plot_cost {plot_cost.min()}')
rowcol_min = np.unravel_index(indices=plot_cost.argmin(), shape=plot_cost.shape)
print(f"Min at {rowcol_min}")
print(f"Min MSE Theta 0 = {plot_t0[111][91]}")
print(f"Min MSE Theta 1 = {plot_t1[111][91]}")

# MSE Gradient Descent


def grad(x, y, thetas):
    n = y.size
    r = y - thetas[0] - thetas[1] * x

    theta0_slope = (-2 * sum(r)) / n
    theta1_slope = (-2 * sum(r * x)) / n

    return np.array([theta0_slope[0], theta1_slope[0]])


multiplier = 0.01
thetas = np.array([2.9, 2.9])

plot_values = thetas.reshape(1, 2)
mse_values = mse1(y=y_5, y_hat=thetas[0] - thetas[1] * x_5)

for _ in range(1000):
    thetas = thetas - multiplier * grad(x=x_5, y=y_5, thetas=thetas)

    plot_values = np.concatenate((plot_values, thetas.reshape(1, 2)), axis=0)
    mse_values = np.append(arr=mse_values, values=mse2(y=y_5, y_hat=thetas[0] + thetas[1] * x_5))

print(f"Min Theta0: {thetas[0]}, min Theta1: {thetas[1]}")
print(mse1(y=y_5, y_hat=thetas[0] + thetas[1] * x_5))

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.scatter(plot_values[:, 0], plot_values[:, 1], mse_values, color="black", s=80)
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha=0.5)
plt.show()
