# -*- coding: utf-8 -*-
"""numpy_tutorial.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JaBXwSC8wuZLKgpovM17eiAU7O2gBUek
"""

import numpy as np
import numpy.linalg as la

"""#Matrices

### Simple matrix
"""

m = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

"""Matrix shape is a tuple storing the number of size of each dimentions.
For 2D matrices, the order is always rows-columns.
"""

m.shape

m

type(m)

n = np.array([
    [[1,2],[3,4]],
    [[1,2],[3,4]]
])
n.shape

"""### Special matrices

#### Identity
"""

np.eye(3, 3)

"""#### Zeros"""

np.zeros((3, 3))

"""#### Ones"""

np.ones((3, 3))

"""## Indexing

For two dimentional matrices, rows are dimention 0 and columns are dimention 1.
"""

m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

m[1, 2]

"""##Slicing

Use the : symbol to get all values across a dimention.
"""

m[1, :]

"""The : can also represent a range"""

m[:, 0:2]

"""## Reshaping

Preserve the data, but change the shape of the matrix.
"""

m = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
m.reshape((2, 3))

"""Reshape a matrix into a single row, figuring out the correct number of columns

"""

m.reshape((1, -1))

f = m.reshape(-1)
f

f.shape

"""## Getting useful statistics

Specify dimension using the axis keyword.

Minimum and maximum values
"""

m

m.max(axis=1)

"""<b>Mean, standard deviation, and variance</b>"""

m.mean(axis=0)

m.std(axis=0)

m.var(axis=0)

"""## Operations on matrices

#### Element-wise
"""

a = np.array([
    [1, 2],
    [3, 4]
])

a + a

a * a

a ** a

"""#### Matrix multiplication"""

np.dot(a,b)

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [1, 1, 1],
    [1, 1, 1]
])
a.dot(b)

"""##Common linear algebra operations


"""

import numpy.linalg as la

"""### Transpose"""

m = np.array([
    [1, 2],
    [3, 4]
])
m.T

"""### Eigenvalues and eigenvectors"""

evals, evects = la.eig(m)

evals

evects

m.dot(evects[:,0])

evals[0] * evects[:,0]

"""### Singular value decomposition"""

U, s, V = np.linalg.svd(m)

"""### Other useful operations
* Determinant: `la.det(m)`
* Norm: `la.norm(m)`
* Inverse: `la.inv(m)`

# Vectors

A vector is a special case of a matrix, and has a single dimension.
"""

np.array([0.1, 0.3, 0.1, 0.5])

"""Comma after length unpacks the first and only element of the tuple into the variable"""

p = np.array([0.1, 0.3, 0.1, 0.5])
length, = p.shape
length

"""## Filtering"""

p > 0.4

p[p > 0.4]

"""## Searching and sorting

"""

p.min()

p.max()

"""Index of the max value in the array"""

p.argmax()

"""Sorting the values"""

p.sort()
p

"""Getting the indeces that correspond to a sorted order of the elements"""

p = np.array([0.1, 0.3, 0.1, 0.5])
p.argsort()

events = np.array(['A', 'B', 'C', 'D'])

"""Suppose we have an array of events, and `p` defines the probability mass function for these event.

To get the two most likely events, we need to figure out the indices of the top two values.

Note: the fancy indexing `[::-1]` reverses the elements in the array using Python slice syntax (start : stop : step).
"""

i = p.argsort()[::-1]
i

"""Getting the top most probable event is easy now:"""

events[i[:2]]

"""# Working with data

###Loading data from a text file

Specify the delimiter: comma, tab, space, etc.
"""

path = 'sample_text'
# Create a text file
with open(path, 'w') as f:
    f.write('1, 2, 3')
np.genfromtxt(path, delimiter=',')

"""`genfromtxt` has a lot of useful optional ([arguments](http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html) )

Allows to specify the datatype to load the data as, specify comment format, skipping headers, etc.

### Loading data from a Matlab file with scipy
"""

from scipy.io import loadmat
data = {}
loadmat(filename, data)

"""`data` is a now dictionary of variable name to data. If the Matlab dump contained a variable `D`, access it by `data['D']`

### Dumping numpy data to file
"""

data = np.array([1, 2, 3])
np.save('numpy_data', data)

"""### Loading numpy data from file"""

data = np.load('numpy_data.npy')
data

m2 = np.arange(9).reshape((3, 3))
m2.argsort()
