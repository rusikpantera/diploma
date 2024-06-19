import math
import random

from sympy import symbols, Eq, solve
from math import exp
import sympy
import numpy as np
import matplotlib.pyplot as plt


def is_complex(x):
    return complex(x).imag != 0


def to_rational(x):
    if 0.01 > complex(x).imag > -00.1:
        return complex(x).real
    return x


def type_define(solution):
    final_solutions = []
    for i in range(len(solution)):

        final_solutions.append(to_rational(solution[i]))

        if is_complex(final_solutions[i]):
            return {'type': 'complex', 'solution': solution}

    return {'type': 'rational', 'solution': tuple([x for x in final_solutions])}


def type_for_all_solutions(solutions):
    s = []
    for solution in solutions:
        s.append(type_define(solution))

    return s


def remove_zeros_solutions(solutions):
    s = []
    for solution in solutions:
        if all(abs(x) > 0.001 for x in solution):
            s.append(solution)
    return s


def solution_system(v1, v2, v3, v4, v5, v6):
    u, v, w = symbols('u v w')

    eq1 = Eq(v3 * u - v2 * v + w * v1 - v4, 0)
    eq2 = Eq(2 * v1 * u * w - v2 * (2 * u * v - 3 * w) + v3 * (2 * u ** 2 - 3 * v) - 3 * v5 + v4 * u, 0)
    eq3 = Eq(-v1 * w ** 2 * u + v2 * u * v * w + v3 * w * (v - u ** 2) + v4 * (u * w - v ** 2) + v5 * u * v + v6 * v, 0)

    equations = [eq1, eq2, eq3]

    unknowns = [u, v, w]

    solutions = solve(equations, unknowns)

    return solutions


def solution_cubic(u, v, w):
    z = symbols('z')

    eq1 = Eq(z ** 3 - u * z ** 2 + v * z - w, 0)

    equations = [eq1]

    unknowns = [z]

    solutions = solve(equations, unknowns)

    return tuple([x[0] for x in solutions])


def solution_c(lambdas, t1, delta, v1, v2, v3):
    c1, c2 = symbols('c1 c2')

    eps1 = Eq(c1 * sympy.exp(lambdas[0] * t1) + c2 * sympy.exp(lambdas[1] * t1), 0)
    eps2 = Eq(
        c1 * sympy.exp(lambdas[0] * (t1 + delta)) + c2 * sympy.exp(lambdas[1] * (t1 + delta)),
        -v1)
    eps3 = Eq(
        c1 * sympy.exp(lambdas[0] * (t1 + 2 * delta)) + c2 * sympy.exp(lambdas[1] * (t1 + 2 * delta)),
        -v1 - v2)

    v1_eq = Eq(eps1.lhs - eps2.lhs, v1)
    v2_eq = Eq(eps2.lhs - eps3.lhs, v2)

    equations = [v1_eq, v2_eq]

    unknowns = [c1, c2]

    solutions = solve(equations, unknowns)

    return solutions


def lambda_solutions(cubic_solution, delta):
    return [math.log(z) / delta for z in cubic_solution]


def result_func(lambdas, c0, c_result, t):
    sum = c0
    for i in range(len(lambdas)):
        sum += c_result[i] * exp(lambdas[i] * t)
    return float(sum)


def get_range(lambdas, c0, c_result, t1, delta, max_value):
    values = [result_func(lambdas, c0, c_result, t1 + delta * i) for i in range(int((max_value - t1) / delta) + 1)]
    x = [t1 + delta * i for i in range(int((max_value - t1) / delta) + 1)]

    return x, values


def plot_values(x_values, y_values, line_style='solid', color='blue', alpha=1.0, markersize=1):
    plt.plot(x_values, y_values, linestyle=line_style, color=color,marker='o', alpha=alpha, markersize=markersize)
    plt.grid(True)


delta = 0.2
t1 = 1
c0 = 5.0
c1 = -2.0
c2 = -0.5
c3 = -0.005
lam1 = -1.0
lam2 = -5.0
lam3 = -10.0
# c0 = 0.749
# c1 = -0.0140
# c2 = -0.00093
# c3 = -0.0000005
# lam1 = -0.7246
# lam2 = -0.7819
# lam3 = -0.85

eps = []
eps.append(c0 + c1 * exp(lam1 * t1) + c2 * exp(lam2 * t1) + c3 * exp(lam3 * t1))
eps.append(c0 + c1 * exp(lam1 * (t1 + delta)) + c2 * exp(lam2 * (t1 + delta)) + c3 * exp(lam3 * (t1 + delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 2 * delta)) + c2 * exp(lam2 * (t1 + 2 * delta)) + c3 * exp(lam3 * (t1 + 2 * delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 3 * delta)) + c2 * exp(lam2 * (t1 + 3 * delta)) + c3 * exp(lam3 * (t1 + 3 * delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 4 * delta)) + c2 * exp(lam2 * (t1 + 4 * delta)) + c3 * exp(lam3 * (t1 + 4 * delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 5 * delta)) + c2 * exp(lam2 * (t1 + 5 * delta)) + c3 * exp(lam3 * (t1 + 5 * delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 6 * delta)) + c2 * exp(lam2 * (t1 + 6 * delta)) + c3 * exp(lam3 * (t1 + 6 * delta)))
eps.append(
    c0 + c1 * exp(lam1 * (t1 + 7 * delta)) + c2 * exp(lam2 * (t1 + 7 * delta)) + c3 * exp(lam3 * (t1 + 7 * delta)))
noise_level = 0.00001
noises = [random.randint(-9, 9) * noise_level for i in range(6)]
print(noises)
v1 = eps[0] - eps[1] + noises[0]
v2 = eps[1] - eps[2] + noises[1]
v3 = eps[2] - eps[3] + noises[2]
v4 = eps[3] - eps[4] + noises[3]
v5 = eps[4] - eps[5] + noises[4]
v6 = eps[5] - eps[6] + noises[5]
print("Direct task")
print("eps1: ", eps[0],
      "eps2: ", eps[1],
      "eps3: ", eps[2],
      "eps4: ", eps[3],
      "eps5: ", eps[4],
      "eps6: ", eps[5],
      "eps7: ", eps[6],
      "eps8: ", eps[7])

print("v1: ", v1,
      "v2: ", v2,
      "v3: ", v3,
      "v4: ", v4,
      "v5: ", v5,
      "v6: ", v6)
print("--------------------")
print("Inverse task")
print("####################")
solutions = solution_system(v1, v2, v3, v4, v5, v6)
solutions = type_for_all_solutions(remove_zeros_solutions(solutions))
solutions = [solution for solution in solutions if solution['type'] == 'rational']
for solution in solutions:
    print("u: ", solution['solution'][0],
          "v: ", solution['solution'][1],
          "w: ", solution['solution'][2])
print("--------------------")
result_l = []
result_c = []
for solution in solutions:

    cubic_solution = solution_cubic(solution['solution'][0], solution['solution'][1], solution['solution'][2])
    cubic_solution = type_define(cubic_solution)

    if cubic_solution['type'] == 'complex':
        continue
    cubic_solution = cubic_solution['solution'][::-1]
    print("Solution u, v, w:")
    print("u: ", solution['solution'][0],
          "v: ", solution['solution'][1],
          "w: ", solution['solution'][2])
    print("--------------------")

    print("z1: ", cubic_solution[0],
          "z2: ", cubic_solution[1],
          "z3: ", cubic_solution[2])
    print("--------------------")

    positive_solution = [x for x in cubic_solution if x > 0]
    print("Positive solutions: ")
    print("z1: ", positive_solution[0],
          "z2: ", positive_solution[1])
    print("--------------------")
    lambdas = lambda_solutions(positive_solution, delta)

    print("Lambdas: ")
    print("lambda1: ", lambdas[0],
          "lambda2: ", lambdas[1])
    print("--------------------")

    sol_c = solution_c(lambdas, t1, delta, v1, v2, v3)
    c_result = [sol_c[x] for x in sol_c]
    print("C: ")
    print("c1: ", c_result[0],
          "c2: ", c_result[1])
    print("--------------------")

    c_0 = 0
    for j in range(len(eps)):
        sum = 0
        for i in range(len(lambdas)):
            sum += c_result[i] * exp(lambdas[i] * (t1 + delta * j))
        c_0 += eps[j] - sum
    c_0 = c_0 / len(eps)
    print("c0: ", c_0)
    print("--------------------")

    result_l.append(lambdas)
    result_c.append([c_0, c_result[0], c_result[1]])
print("END INVERSE TASK")
print("--------------------")

print("Actual values")
print("c0: ", c0,
      "c1: ", c1,
      "c2: ", c2,
      "c3: ", c3)
print("lambda1: ", lam1,
      "lambda2: ", lam2,
      "lambda3: ", lam3)
print("--------------------")
print("Restored values")

result_l = np.array(result_l).mean(axis=0)
result_c = np.array(result_c).mean(axis=0)
print("c0: ", result_c[0],
      "c1: ", result_c[1],
      "c2: ", result_c[2]
      )
print("lambda1: ", result_l[0],
      "lambda2: ", result_l[1])
print("--------------------")

x, y = get_range([lam1, lam2, lam3], c0, [c1, c2, c3], t1, delta, 10)
plot_values(x, y, color='#009B95', alpha=1)
x, y = get_range(result_l, result_c[0], result_c[1:], t1, delta, 10)
plot_values(x, y, line_style='', color='black', markersize=3.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('График функций')
plt.grid(True)
plt.legend(['Точный', 'Восстановленный'])
plt.show()
