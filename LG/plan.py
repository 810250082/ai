# -*-coding:utf-8-*-

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import random

fig = figure()
ax = Axes3D(fig)

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
Z = 3*X + 4*Y
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
show()

def plan(x, y, z):
    a = 3*x + 4*y - z
    if a > 0:
        return 1
    else:
        return 0


def random_x(a=100, b=100):
    return round(random.uniform(-100, 100), 2)


def product_data():
    data = {'positive': [], 'nagetive': []}
    while len(data['positive']) != 50 or len(data['nagetive']) !=50:
        x, y, z = random_x(), random_x(), random_x()
        if plan(x, y, z) and len(data['positive']) < 50:
            data['positive'].append((x, y, z, 1))
        elif not plan(x, y, z) and len(data['nagetive']) < 50:
            data['nagetive'].append((x, y, z, 0))
    examples = []
    examples.extend(data['positive'])
    examples.extend(data['nagetive'])
    random.shuffle(examples)
    examples_str = []
    for item in examples:
        examples_str.append('{0[0]}\t{0[1]}\t{0[2]}\t{0[3]}'.format(item))
    examples_str = '\n'.join(examples_str)
    with open('datas1', 'w') as f:
        f.write(examples_str)

product_data()
b = 1