
import math
import numpy as np
from itertools import chain


class TestClass:
    def __init_(self):
        instance_var = 2


def test_itertools():
    a = np.array([[1,2],[3,4],[5,6]])
    b = np.arange(24).reshape(4, 3, 2)
    flat_iterator = chain.from_iterable(b)
    particles_to_check = np.fromiter(flat_iterator, dtype=np.intp)
    print(particles_to_check)


def test_class_indexing():
    test_instance = TestClass()
    # test_instance[2]


def test_ceil():
    a = math.ceil(12.2)
    print()
    print(a)
    print(type(a))


def main():
    # test_itertools()
    # test_class_indexing()
    test_ceil()


if __name__ == "__main__":
    main()
