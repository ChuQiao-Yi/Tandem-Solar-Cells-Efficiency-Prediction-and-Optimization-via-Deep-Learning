weight = [1, 3, 1, 14] # weight
lb = [1, 1.5, 16, 1, 1]
ub = [10, 8, 25, 5, 5] #boundaries
T=None # starting temperature, recommendation is more than 100
T_min=None # ending temperature, must lower enough for iterating than 500 iterations. recommendation is 1e-250
q=None # decline of temperature, recommendation is 0.8 to 0.99
L=None # the number of iterations for each cooling down, recommendation is 200 to 500
max_iter=None # max iteration, identical with GA, PSO, recommendation is 500