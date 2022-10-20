# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\02450Toolbox_Python\Scripts"
# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\project1"
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import scipy.linalg as linalg
import array_to_latex as a2l
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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

model = lm.LinearRegression()
model.fit(X_normalized, y)


data = np.concatenate((X_normalized, y[:,np.newaxis]), axis=1)

alphas = 10 ** np.linspace(-4,8,13)

'''
reg_models = []
for i in range(len(alphas)):
    reg_models.append( lm.Ridge(alpha=alphas[i]) )
    reg_models[i].fit(X_normalized, y)
    print(reg_models[i].coef_)

kf = KFold(n_splits=3)
tmp = list(kf.split(X_normalized))
'''


def train_model(data, train_idx, alpha):
    model = lm.Ridge(alpha=alpha)
    model.fit(data[train_idx,:-1], data[train_idx,-1])
    return model

def test_model(data, test_idx, model):
    y_pred = model.predict(data[test_idx,:-1])
    return mean_squared_error(data[test_idx,-1], y_pred)

def lm_gen_error(data, alpha):
    kf = KFold(n_splits=10)
    test_error = []
    for train_idx, test_idx in kf.split(data):
        model = train_model(data, train_idx, alpha)
        test_error.append(test_model(data, test_idx, model))
    return sum(test_error)/len(test_error)

lm_gen_errors = np.empty((len(alphas)))
for i in range(len(alphas)):
    lm_gen_errors[i] = lm_gen_error(data, alphas[i])

#plt.scatter(np.log10(alphas), lm_gen_errors)

min_idx = np.argmin(lm_gen_errors)
print(f"({alphas[min_idx]}, {lm_gen_errors[min_idx]})")


# optimal model
model = lm.Ridge(alpha=alphas[min_idx])
model.fit(data[:,:-1], data[:,-1])
print(model.coef_)



### baseline
def base_gen_error(data):
    kf = KFold(n_splits=10)
    test_error = []
    for train_idx, test_idx in kf.split(data):
        train_mean = np.mean(data[train_idx,-1])
        test_error.append( mean_squared_error( data[test_idx,-1], np.full( (len(test_idx)), train_mean) ) )
    return sum(test_error)/len(test_error)

print(base_gen_error(data))

### ANN



### two-level CV
K1, K2 = 10, 10
outer_kf = KFold(n_splits=K1)
E_val = np.empty((len(alphas), K2))
E_test = np.empty((K1))

outer_count = -1
for par_idx, test_idx in outer_kf.split(data):
    outer_count += 1
    inner_kf = KFold(n_splits=K2)
    inner_count = -1
    for train_idx, val_idx in inner_kf.split(par_idx):
        inner_count += 1
        for s in range(len(alphas)):
            model = train_model(data[par_idx], train_idx, alphas[s])
            E_val[s,inner_count] = test_model(data[par_idx], val_idx, model)
    E_gen = np.mean(E_val, axis=1)
    min_idx = np.argmin(E_gen)
    model_star = train_model(data, par_idx, alphas[min_idx])
    E_test[outer_count] = test_model(data, test_idx, model_star)
    print(f"{round(100*(outer_count+1)/K1)}%")
print(np.mean(E_test))

'''
outer_splits = list(outer_kf.split(data))
penisring = outer_splits[3][0]
inner_kf = KFold(n_splits=K2)
dildo = list(inner_kf.split(penisring))
'''


### logistic regression
# binarize y-values

y_bin = np.zeros(y.shape)
y_bin[y > np.mean(y)] = 1

data_bin = np.concatenate((X_normalized, y_bin[:,np.newaxis]), axis=1)

def bin_train_model(data, train_idx, alpha):
    model = lm.LogisticRegression(random_state=0, C=alpha)
    model.fit(data[train_idx,:-1], data[train_idx,-1])
    return model

def bin_test_model(data, test_idx, model):
    y_pred = model.predict(data[test_idx,:-1])
    return np.sum(data[test_idx,-1] != y_pred) / test_idx.shape[0]

# two-level CV
K1, K2 = 10, 10
outer_kf = KFold(n_splits=K1)
E_val = np.empty((len(alphas), K2))
E_test = np.empty((K1))

outer_count = -1
for par_idx, test_idx in outer_kf.split(data_bin):
    outer_count += 1
    inner_kf = KFold(n_splits=K2)
    inner_count = -1
    for train_idx, val_idx in inner_kf.split(par_idx):
        inner_count += 1
        for s in range(len(alphas)):
            model = bin_train_model(data_bin[par_idx], train_idx, alphas[s])
            E_val[s,inner_count] = bin_test_model(data_bin[par_idx], val_idx, model)
    E_gen = np.mean(E_val, axis=1)
    min_idx = np.argmin(E_gen)
    model_star = bin_train_model(data_bin, par_idx, alphas[min_idx])
    E_test[outer_count] = bin_test_model(data_bin, test_idx, model_star)
    print(f"{round(100*(outer_count+1)/K1)}%")
print(np.mean(E_test))



