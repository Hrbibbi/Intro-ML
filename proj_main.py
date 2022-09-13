# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\02450Toolbox_Python\Scripts"
# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\project1"
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import scipy.linalg as linalg
# Load data from excel to np.array
doc = xlrd.open_workbook('./Concrete_Data.xls').sheet_by_index(0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=9)

X = np.empty((1030,8))
for i in range(8):
    X[:,i] = np.array(doc.col_values(i,1,1031)).T

y = np.array(doc.col_values(8,1,1031)).T

# Normalized data
N, M = X.shape

#plt.hist(X[:,7], bins=30)
#plt.show
X_normalized = (X - np.ones((N,1))*X.mean(axis=0)) / (np.ones((N,1))*np.std(X, axis=0))


plt.hist(X_normalized[:,1], bins=30)
plt.show

U,S,V = linalg.svd(X_normalized,full_matrices=False)

sq_sum = np.sum(S*S)
running_sum = 0
PCA_weight = []
for i in range(8):
    running_sum += S[i]**2
    PCA_weight.append(running_sum / sq_sum)

V[0,:]

# Plot PCA of the data
Z = X @ V[:2,:].T
plt.scatter(Z[:,0], Z[:,1])

##Summary Statistics of the dataset
Mean=[]
Variance=[]
Standard_Deviation=[]
for i in range(8):
    Mean.append(X[:,i].mean())
    Variance.append(X[:,i].var())
    Standard_Deviation.append(X[:,i].std())

COVariance=np.cov(X.T)
Correlation=np.corrcoef(X.T)
