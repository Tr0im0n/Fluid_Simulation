
import numpy as np
from itertools import chain


a = np.array([[1,2],[3,4],[5,6]])

b = np.arange(24).reshape(4, 3, 2)

flat_iterator = chain.from_iterable(b)

particles_to_check = np.fromiter(flat_iterator, dtype=np.intp)

print(particles_to_check)


