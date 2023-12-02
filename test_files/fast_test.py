import random
import math

"""
Example of stochastic gradient
"""

def test_I(I, h, l, demands):
    ris = 0
    for d in demands:
        ris += h * max(I-d, 0) + l * max(d-I, 0)
    return ris

def stochastic_gradient(h, l, demands):
    I = 10 # invece di partire subito magari aspetto un po' e prendo il d_max osservato
    for i, d in enumerate(demands):
        grad = 0
        if I > d:
            grad = h
        else:   
            grad = -l
        I = I - (1/math.log(i+2))* grad
    return I


demands = random.choices(
    population=[1, 2, 3, 4, 5],
    weights=[3, 1, 1, 1, 2],
    k=100
)

l = 1
h = 0.5

# BRUTE FORCE
for i in range(5+2):
    print(i, test_I(i, h, l, demands))



print(stochastic_gradient(h, l, demands))
