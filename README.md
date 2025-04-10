# integral-equation-solver
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

def kernel_function(x, s):
    """ Example kernel function K(x, s) """
    return np.exp(-np.abs(x - s))

def function_f(x):
    """ Right-hand side function f(x) """
    return np.sin(x)

def integral_equation_solver(a, b, num_points=100):
    """ Solves the integral equation of the form:
         g(x) = f(x) + \int_a^b K(x, s) g(s) ds
    """
    x_values = np.linspace(a, b, num_points)
    
    def g_approximation(g_values, x):
        integral_result, _ = spi.quad(lambda s: kernel_function(x, s) * np.interp(s, x_values, g_values), a, b)
        return function_f(x) + integral_result
    
    # Initial guess (f(x) as approximation for g(x))
    g_values = np.array([function_f(x) for x in x_values])
    
    # Iterative solving
    for _ in range(10):  # Adjust the number of iterations as needed
        g_values = np.array([g_approximation(g_values, x) for x in x_values])
    
    return x_values, g_values

def plot_solution(a, b):
    """ Plots the solution of the integral equation """
    x_values, g_values = integral_equation_solver(a, b)
    plt.plot(x_values, g_values, label='Solution g(x)', color='blue')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title('Solution of Integral Equation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_solution(0, 5)
wsderf
