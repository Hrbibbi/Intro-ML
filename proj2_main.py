# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\02450Toolbox_Python\Scripts"
# cd "C:\Users\diama\OneDrive - Danmarks Tekniske Universitet\02450_ml\project1"
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import scipy.linalg as linalg
#import array_to_latex as a2l
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from scipy import stats
import torch


from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary


# Load data from excel to np.array
doc = xlrd.open_workbook('./Concrete_Data.xls').sheet_by_index(0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=9)

X = np.empty((1030,8))
for i in range(8):
    X[:,i] = np.array(doc.col_values(i,1,1031)).T

y = np.array(doc.col_values(8,1,1031)).T
#y = np.expand_dims(y,axis=1)

# Normalized data
N, M = X.shape

#plt.hist(X[:,7], bins=30)
#plt.show
X_normalized = (X - np.ones((N,1))*X.mean(axis=0)) / (np.ones((N,1))*np.std(X, axis=0))

model = lm.LinearRegression()
model.fit(X_normalized, y)

data = np.concatenate((X_normalized, y[:,np.newaxis]), axis=1)

alphas = 10 ** np.linspace(-4,8,13)

the_table = np.zeros((10,6))
the_table[:,0] = np.arange(10) + 1

reg_models = []
for i in range(len(alphas)):
    reg_models.append( lm.Ridge(alpha=alphas[i]) )
    reg_models[i].fit(X_normalized, y)
    print(reg_models[i].coef_)

kf = KFold(n_splits=3)
tmp = list(kf.split(X_normalized))



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
    return test_error

base_test_errors = base_gen_error(data)
the_table[:,5] = base_test_errors


### linear two-level CV
K1, K2 = 10, 10
outer_kf = KFold(n_splits=K1)
E_val = np.empty((len(alphas), K2))
E_test = np.empty((K1))
min_alphas = np.empty((K1))
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
    min_alphas[outer_count] = alphas[min_idx]
    model_star = train_model(data, par_idx, alphas[min_idx])
    E_test[outer_count] = test_model(data, test_idx, model_star)
    print(f"{round(100*(outer_count+1)/K1)}%")
print(np.mean(E_test))

the_table[:,3] = min_alphas
the_table[:,4] = E_test


outer_splits = list(outer_kf.split(data))
penisring = outer_splits[3][0]
inner_kf = KFold(n_splits=K2)
dildo = list(inner_kf.split(penisring))


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




### ANN
def ANN_train_model(data, train_idx, h_units):
    n_hidden_units = h_units     # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(data[train_idx,:-1])
    y_train = torch.Tensor(np.expand_dims(data[train_idx,-1],axis=1))
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    return net

def ANN_test_model(data, test_idx, net):
    X_test = torch.Tensor(data[test_idx,:-1])
    y_test = torch.Tensor(np.expand_dims(data[test_idx,-1],axis=1))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    return (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean # MSE

ANN_model = ANN_train_model(data, np.arange(data.shape[0]), 5)

ANN_test_model(data, np.arange(data.shape[0]), ANN_model)
'''

### two-level CV
Hs = np.array([1,5,10,15,20])
K1, K2 = 10, 10
outer_kf = KFold(n_splits=K1)
E_val = np.empty((len(Hs), K2))
E_test = np.empty((K1))
min_Hs = np.empty((K1))
outer_count = -1
for par_idx, test_idx in outer_kf.split(data):
    outer_count += 1
    inner_kf = KFold(n_splits=K2)
    inner_count = -1
    for train_idx, val_idx in inner_kf.split(par_idx):
        inner_count += 1
        for s in range(len(Hs)):
            model = ANN_train_model(data[par_idx], train_idx, Hs[s])
            E_val[s,inner_count] = ANN_test_model(data[par_idx], val_idx, model)
    E_gen = np.mean(E_val, axis=1)
    min_idx = np.argmin(E_gen)
    min_Hs[outer_count] = Hs[min_idx]
    model_star = ANN_train_model(data, par_idx, Hs[min_idx])
    E_test[outer_count] = ANN_test_model(data, test_idx, model_star)
    print(f"{round(100*(outer_count+1)/K1)}%")
print(np.mean(E_test))


ANN_data_arr = np.concatenate( (min_Hs[:,np.newaxis], E_test[:,np.newaxis]), axis=1)
np.savetxt("ANN_data", ANN_data_arr)


the_table[:,1] = min_Hs
the_table[:,2] = E_test
'''

from scipy.stats import t

nu = 10-1
r = np.empty((10))
K = KFold(n_splits=10, shuffle=True)
count = -1
for train_idx, test_idx in K.split(data):
    count += 1
    lin_model = train_model(data, train_idx, 1)
    net_model = ANN_train_model(data, train_idx, 2)
    r[count] = test_model(data, test_idx, lin_model) - ANN_test_model(data, test_idx, net_model)
r_hat = np.mean(r)
s_hat = np.std(r)
s_tilde = np.sqrt( (1/10 + 1/(10-1)) ) * s_hat

sig_level = 0.05
conf_int = [t.ppf(sig_level/2, nu), t.ppf(1-sig_level/2, nu)]
t_hat = r_hat/s_tilde

t.cdf(t_hat, nu)


#binary ANN
def ANN_train_model_b(data, train_idx, h_units):
    n_hidden_units = h_units     # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                        # 1st transfer function, either Tanh or ReLU:
                        torch.nn.Tanh(),                            #torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                        torch.nn.Sigmoid() # final tranfer function
                        )
    # Since we're training a neural network for binary classification, we use a 
    # binary cross entropy loss (see the help(train_neural_net) for more on
    # the loss_fn input to the function)
    loss_fn = torch.nn.BCELoss()
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(data[train_idx,:-1])
    y_train = torch.Tensor(np.expand_dims(data[train_idx,-1],axis=1))
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    return net
def ANN_test_model_b(data, test_idx, net):
    X_test = torch.Tensor(data[test_idx,:-1])
    y_test = torch.Tensor(np.expand_dims(data[test_idx,-1],axis=1))
     
    # Determine errors and errors
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    return error_rate


### two-level CV
Hs = np.array([1,5,10,15,20])
K1, K2 = 10,10
outer_kf = KFold(n_splits=K1)
E_val = np.empty((len(Hs), K2))
E_test = np.empty((K1))
min_Hs = np.empty((K1))
outer_count = -1
for par_idx, test_idx in outer_kf.split(data_bin):
    outer_count += 1
    inner_kf = KFold(n_splits=K2)
    inner_count = -1
    for train_idx, val_idx in inner_kf.split(par_idx):
        inner_count += 1
        for s in range(len(Hs)):
            model = ANN_train_model_b(data_bin[par_idx], train_idx, Hs[s])
            E_val[s,inner_count] = ANN_test_model_b(data_bin[par_idx], val_idx, model)
    E_gen = np.mean(E_val, axis=1)
    min_idx = np.argmin(E_gen)
    min_Hs[outer_count] = Hs[min_idx]
    model_star = ANN_train_model_b(data_bin, par_idx, Hs[min_idx])
    E_test[outer_count] = ANN_test_model_b(data_bin, test_idx, model_star)
    print(f"{round(100*(outer_count+1)/K1)}%")
print(np.mean(E_test))