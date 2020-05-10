import numpy as np


def kmor(X: np.array, k: int, y: float = 3, nc0: float = 0.1, max_iteration: int = 100, gamma: float = 10 ** -6):
    """K-means clustering with outlier removal

    Parameters
    ----------
    X
        Your data.
    k
        Number of clusters.
    y
        Parameter for outlier detection. Increase this to make outlier removal subtle.
    nc0
        Maximum percentage of your data that can be assigned to outlier cluster.
    max_iteration
        Maximum number of iterations.
    gamma
        Used to check the convergence.

    Returns
    -------
    numpy.array
        Numpy array that contains the assigned cluster of each data point (0 to k, the cluster k is the outlier
        cluster)
    """
    n = X.shape[0]
    n0 = int(nc0 * X.shape[0])
    Z = X[np.random.choice(n, k)]

    def calculate_dd(U, Z):
        return np.linalg.norm(X - Z[U], axis=1) ** 2

    def calculate_D(outliers, dd):
        factor = y / (n - outliers.size)
        return factor * np.sum(np.delete(dd, outliers))

    def calculate_U(X):
        def closest(p):
            return np.argmin(np.linalg.norm(Z - p, axis=1))

        return np.apply_along_axis(closest, 1, X)

    outliers = np.array([])
    U = calculate_U(X)

    s = 0
    p = 0

    while True:
        # Update U (Theorem 1)
        dd = calculate_dd(U, Z)
        D = calculate_D(outliers, dd)

        dd2 = dd[dd > D]
        outliers = np.arange(n)[dd > D][dd2.argsort()[::-1]]
        outliers = outliers[:n0]

        U = calculate_U(X)

        # Update Z (Theorem 3)
        is_outlier = np.isin(U, outliers)

        def mean_group(i):
            x = X[np.logical_and(U == i, ~is_outlier)]
            # Empty group
            if x.size == 0:
                x = X[np.random.choice(n, 1)]
            return x.mean(axis=0)

        Z = np.array([mean_group(i) for i in range(k)])

        # Update P
        dd = calculate_dd(U, Z)
        D = calculate_D(outliers, dd)
        if outliers.size == 0:
            p1 = np.sum(dd)
        else:
            p1 = np.sum(dd[~outliers]) + D * outliers.size

        # Exit condition
        s += 1
        if abs(p1 - p) < gamma or s > max_iteration:
            break

        p = p1
        print("s:", s, "p:", p)

    U[outliers] = k
    return U
