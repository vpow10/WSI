from solver import GradientIndent
import numpy as np
import matplotlib.pyplot as plt


# functions and gradients

f = lambda x: 2*x**2 + 3*x - 1
f_gradient = lambda x: 4*x + 3
g = lambda x1, x2: 1 - 0.6*np.exp(- x1**2 - x2**2 ) - 0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
g_gradient = lambda x1, x2: (1.2*x1*np.exp(- x1**2 - x2**2) + 0.8*(x1 + 1.75)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2),
                              1.2*x2*np.exp(- x1**2 - x2**2) + 0.8*(x2 - 1)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2))
epsilon = 1e-6


print(g(0.01, 0.02))
print(g(-1.69, 0.97))
# testing f function

betas = np.arange(0.02, 0.49, 0.01)
its = []

for beta in betas:
    x0 = np.random.uniform(-100, 100, 1)
    solver = GradientIndent(beta)
    it = solver.solver(f, f_gradient, epsilon, x0, show_data=False)[1]
    its.append(it)

fig1 = plt.figure("Function f beta - iterations")
plt.bar(betas, its, color='purple', width=0.006)
plt.xlim(betas[0] - 0.007/2, betas[-1] + 0.008/2)
plt.xticks(betas[::4])
plt.xlabel('beta')
plt.ylabel('iterations')
plt.title("Function f beta - iterations relation")
f_beta_min = round(betas[np.argmin(its)], 3)

# testing best beta

solver = GradientIndent(f_beta_min)
x0s = []
mins = []
its = []
for x in range(200):
    x0s.append(np.random.uniform(-1e6, 1e6, 1))
for x0 in x0s:
    (minim, it) = solver.solver(f, f_gradient, epsilon, x0, show_data=False)
    mins.append(minim)
    its.append(it)
fig2 = plt.figure("Function f best beta")
ax = fig2.add_subplot(projection='3d')
ax.scatter(x0s, mins, its, c='purple')
ax.set_xlabel('x0')
ax.set_ylabel('minimum')
ax.set_zlabel('iterations')

# testing g function

betas = np.arange(0.2, 1.5, 0.03)
its = []

for beta in betas:
    x0 = np.random.uniform(-2.5, 2.5, (100, 2))
    solver = GradientIndent(beta)
    it = [solver.solver(g, g_gradient, epsilon, x1, x2, show_data=False)[1] for x1, x2 in x0]
    its.append(np.median(it))

fig3 = plt.figure("Function g beta - iterations")
plt.bar(betas, its, color='purple', width=0.025)
plt.xlim(betas[0] - 0.0225/2, betas[-1] + 0.0275/2)
plt.xticks(betas[::4])
plt.xlabel('beta')
plt.ylabel('iterations')
plt.title("Function g beta - iterations relation")
g_beta_min = round(betas[np.argmin(its)], 3)

# testing best beta

solver = GradientIndent(g_beta_min)
x0s = []
mins = []
its = []
for x in np.random.uniform(-5, 5, (100, 2)):
    x0s.append(x)
    minim, it = solver.solver(g, g_gradient, epsilon, x[0], x[1], show_data=False)
    mins.append(minim)
    its.append(it)
for n in range(len(its)):
    if its[n] > 2500:
        its[n] = 2500
fig4 = plt.figure("Function g best beta")
plt.scatter(np.transpose(x0s)[0], np.transpose(x0s)[1], its, cmap='viridis', alpha=0.75, label='Main points')
plt.scatter(np.transpose(mins)[0], np.transpose(mins)[1], marker='x', c='red', label='Minimum points')

for i in range(len(x0s)):
    plt.plot([x0s[i][0], mins[i][0]], [x0s[i][1], mins[i][1]], 'k--', alpha=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Function g best beta')
plt.legend()
plt.grid(True)

plt.show()