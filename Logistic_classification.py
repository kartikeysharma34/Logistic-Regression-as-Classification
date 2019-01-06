import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import numpy.random as nr
import math
from sklearn import preprocessing

#Logistic Regression PLot

xseq = np.arange(-7, 7, 0.1)

logistic = [math.exp(v)/(1+math.exp(v)) for v in xseq]
plt.plot(xseq, logistic, color = 'red')
plt.plot([-7,7], [0.5, 0.5], color = 'blue')
plt.plot([0, 0], [0,1], color = 'blue')
plt.title('Logistic function for two-class classification')
plt.xlabel('log likelihood')
plt.ylabel('Value of output from Linear regression')
plt.show()

