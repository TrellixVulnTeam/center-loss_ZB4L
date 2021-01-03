# test mnist tsne visualize
from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

mnist = fetch_openml("MNIST original")
X = mnist.data / 255.0
y = mnist.target
print(X.shape, y.shape)
