import numpy as np
import matplotlib.pyplot as plt


# f function

f = lambda x: 2*x**2 + 3*x - 1
x = np.arange(-3, 3, 0.1)
y = f(x)
fig1 = plt.figure("Function f")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)

# g function

g = lambda x1, x2: 1-0.6*np.exp(- x1**2 - x2**2 )-0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
x1 = np.arange(-3,3,0.1)
x2 = np.arange(-3,3,0.1)
x1, x2 = np.meshgrid(x1, x2)
z = g(x1, x2)

# g function figure 1

fig2 = plt.figure('Function g picture 1')
plt.xlabel('x1')
plt.ylabel('x2')
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

# g function figure 2

fig3 = plt.figure('Function g picture 2')
plt.contourf(x1, x2, z, 15)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')


plt.show()