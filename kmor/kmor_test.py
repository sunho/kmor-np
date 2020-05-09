from kmor import kmor
import numpy as np

def test_simple_kmor():
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [3, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 10]
    ])
    U = kmor(X, 1)
    np.testing.assert_equal(U, np.array([0]*9 + [1]))
