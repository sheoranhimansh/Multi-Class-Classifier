
# coding: utf-8

# In[1]:


from __future__ import division
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import sys  
import math


# In[2]:
path_train_file = sys.argv[1]
path_test_file = sys.argv[2]

constants = dict()
constants['lr'] = 1e-3
constants['reg'] = 500
constants['num_iters'] = 10000
constants['const2'] = 1e50
constants['const' ] = 0.001
# print(constants['lr'])
    

# In[3]:


input_file = open(path_train_file, "r")
a = []
a_name = []
name_classes = []
for file in input_file:
    arr = np.array(Image.open(file.split()[0]).convert("L").resize((32,32))).flatten()
    a.append(arr)
    a_name.append(str(file.split()[1]))
    name_classes.append(str(file.split()[1]))
a_name = np.array(a_name)
unique_classes = list(set(a_name))


# In[4]:


x_files = open(path_test_file, "r")
file_name = []  
res = []
for file in x_files:
    file = file.split('\n')
    f = Image.open(str(file[0]))
    f = f.convert("L")
    f = f.resize((32,32))
    file_name.append(str(file[0]))
    arr = np.asarray(f)
    arr = arr.reshape(-1)
    res.append(arr)
    f.close


# In[5]:
def matmul(a,b):
    ans = np.matmul(a,b)
    return ans


a = np.array(a)
mean = np.mean(a,axis=0)
mean1 = a - mean
v, eigenvalues , eigenvectors = np.linalg.svd(mean1)
eigenvectors = eigenvectors.T
coefficiets = matmul(mean1 , eigenvectors) 
eigenvectors = eigenvectors.T
arr1 = np.array(coefficiets[:,:32])
b = np.ones((len(arr1),1))
arr1 = np.hstack((arr1,b))
new_img = np.array(arr1)
eigenvectors = eigenvectors[0:32,:]
eigenvectors = eigenvectors.T


# In[6]:





# In[7]:


assign_int = dict()
assign_name = dict()
rang = len(unique_classes)
for i in range(rang):
    assign_int[i] = unique_classes[i]
    assign_name[unique_classes[i]] = i
    
rang  = len(a_name)
for i in range(rang):
    a_name[i] = assign_name[a_name[i]]
a_name = a_name.astype(int)


# In[8]:


num_classes = len(set(a_name))
newt = new_img.T
X = np.array(newt)
D = len(X)
W = np.random.randn(num_classes, D) *constants['const' ]
Y = np.array(a_name)


# In[9]:
def maxf(a):
    ans = np.max(a)
    return ans
def exp(a):
    ans = np.exp(a)
    return ans
def log(a):
    ans = np.log(a)
    return ans
losses_history = []
for i in range(constants['num_iters']):
    loss = 0 
    grad = np.zeros_like(W)
    dim  = X.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)
    scores -= maxf(scores)
    scores_exp = exp(scores)
    scores_exp = scores_exp*constants['const2']
    rang = range(num_train)
    correct_scores_exp = scores_exp[Y, rang]
    scores_exp_sum = np.sum(scores_exp, axis=0)
    frac = correct_scores_exp / scores_exp_sum
    loss = -np.sum(log(frac))
    loss /= num_train
    scores_exp_normalized = scores_exp / scores_exp_sum
    rang = range(num_train)
    scores_exp_normalized[Y, rang] -= 1
    XT = X.T
    grad = scores_exp_normalized.dot(X.T)
    grad /= num_train
    losses_history.append(loss)
    prod = constants['lr'] * grad
    W -= prod
    if(loss<1):
        break;
    


# In[10]:


res = res - mean
coefficiets = matmul(res , eigenvectors)
len1 = len(res)
b = np.ones((len1,1))
res = np.hstack((coefficiets,b))
array1 = np.array(res).T
shap = array1.shape[1]
pred_ys = np.zeros(shap)
scores = W.dot(array1)
pred_ys = np.argmax(scores, axis=0)
pred = pred_ys


# In[14]:
# print(pred)
for i in range(len(pred)):
     print(assign_int[pred[i]])