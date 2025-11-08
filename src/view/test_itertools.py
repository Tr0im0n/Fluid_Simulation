
import math
import numpy as np
import pygame
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
    

def test_font():
    pygame.font.init()
    available_fonts = pygame.font.get_fonts()
    print(available_fonts)
    
    
def test_colormap():
    from src.utils.colors import colormap_array_BWR

    color = colormap_array_BWR(np.zeros((2, 2), dtype=np.float32))
    print(color)


def test_array_indexing():
    a = np.arange(6)
    a.shape = (3, 2)
    print(a)
    print(a[0, 1])
    

def test_np_type():
    a = np.array([1, 2], dtype=np.float32)
    b = a[0]
    c = b.astype(np.int32)
    print(type(c))
    
    
def test_np_concatenate():
    a = [[], [1], [2, 3]]
    b = np.concatenate(a)
    print(b)
    
    
def test_nan():
    a = np.array([1, 2, 3, np.nan, np.nan])
    b = np.square(a)
    print(b)
    
    
def test_reference():
    a = [1, 2]
    b = a
    a = 1
    print(b)


def main():
    # test_itertools()
    # test_class_indexing()
    # test_ceil()
    # test_font()
    # test_colormap()
    # test_array_indexing()
    # test_np_type()
    # test_np_concatenate()
    # test_nan()
    test_reference()
    

if __name__ == "__main__":
    main()
