# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\02450Toolbox_Python\Scripts"
# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\project1"
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import scipy.linalg as linalg
import array_to_latex as a2l
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


##
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('PCA_variance_explained', dpi=300)
plt.show()
##

# Plot PCA of the data
#Z = X @ V[:3,:].T
Z = X_normalized @ V[:3,:].T
#plt.scatter(Z[:,0], Z[:,1], c=y, cmap='Reds')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=y, cmap='Reds')
plt.xlabel(r'$v_1$')
plt.ylabel(r'$v_2$')
plt.title(r'$v_3$')


plt.savefig('PCA_3D_projection', dpi=300)
plt.show()
