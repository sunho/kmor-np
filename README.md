# KMOR Numpy

The python implementation for k-means clustering with outlier removal from the paper written by Guojun Gan et al. [[1]](#1)

## Installation

```
pip install kmor
```

You can also install by conda

```
conda install -c ksunhokim kmor
```

## Example

```python
import numpy as np
from kmor import kmor
X = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,0,100]
])
U = kmor(X, 1)
print(U) # [0,0,0,1]
```

The outliers are assigned to the extra cluster k.

## Documentation

kmor(X, k, y, nc0, max_iteration, gamma)

| Parameter     | Description                                                                    |
|---------------|--------------------------------------------------------------------------------|
| X             | Your data.                                                                     |
| k             | Number of clusters.                                                            |
| y             | Parameter for outlier detection. (default=3) Increase this to make outlier removal subtle. |
| nc0           | Maximum percentage of your data that can be assigned to outlier cluster. (default=0.1)    |
| max_iteration | Maximum number of iterations.                                                  |
| gamma         | Used to check the convergence.                                                 |


## References

<a id="1">[1]</a> Gan, Guojun, and Michael Kwok-Po Ng. "K-means clustering with outlier removal." Pattern Recognition Letters 90 (2017): 8-14.