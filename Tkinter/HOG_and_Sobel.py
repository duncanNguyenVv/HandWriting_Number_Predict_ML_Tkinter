import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from skimage.feature import hog
from skimage.filters import roberts, sobel
import warnings
def extract_hog(img, img_size=28, orientations=18, pixels_per_cell=(4, 4), cells_per_block=(3, 3)):
  fd = hog(img.reshape((img_size, img_size)), orientations=orientations, pixels_per_cell=pixels_per_cell, 
              cells_per_block=cells_per_block)
  return fd
def edges_extraction(X, image_shape=(28,28)):
  fd_list = []
  for row in X:
      img = row.reshape(image_shape)
      fd = sobel(img.reshape(28,28))
      fd_list.append(fd)
  
  return np.array(fd_list)
class HOG_and_Sobel(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass
    
    def fit(self, X, y=None):
      # find pixel 0  or 255
      #self.idx = np.argwhere(np.all(X[..., :] == 0, axis=0))
      return self
    
    def transform(self, X, y=None):
      X_ = X.copy()
      X_ = Parallel(n_jobs=os.cpu_count())(delayed(extract_hog)(img) for img in tqdm(X_))
      X_ = np.array(X_)
      #Todo sobel
      X_sobel = X.copy()
      X_sobel = edges_extraction(X_sobel)
      X_sobel = Parallel(n_jobs=os.cpu_count())(delayed(extract_hog)(img) for img in tqdm(X_sobel))
      X_sobel = np.array(X_sobel)
      #concat
      X_ = np.concatenate((X_, X_sobel), axis=1)
      return X_