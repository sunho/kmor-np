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

## References

<a id="1">[1]</a> Gan, Guojun, and Michael Kwok-Po Ng. "K-means clustering with outlier removal." Pattern Recognition Letters 90 (2017): 8-14.