
# coding: utf-8

# In[1]:


import os
import sys
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import math


# In[2]:
path_train_file = sys.argv[1]
path_test_file = sys.argv[2]

def svd(a):
    ans = np.linalg.svd(a)
    return ans


def calc_prob(x,mean,stdev,y):
    xmean = x-mean
    stdev1 = pow(stdev,2)
    sigma = 2*stdev1
    xmeansq = pow(xmean,2)
    frac = (xmeansq)/(sigma)
    frac = (-1)*frac
    exponent = exp(frac)
    constant = sqrt(2*math.pi)
    constant = constant*stdev
    ans = 1/constant
    ans = ans*exponent
    return ans


# In[3]:

def openf(a):
    return open(a,'r')

def asarray(a):
    ans = np.asarray(a)
    return ans

input_file = openf(path_train_file)
X = []
X_name = []
for file in input_file:
    size =  (32,32)
    im = Image.open(file.split()[0]).convert("L").resize(size)
    array = asarray(im)
    X.append(array.reshape([-1]))
    name = str(file.split()[1])
    X_name.append(name)
    im.close
X = np.array(X)
X_name = np.array(X_name)


# In[4]:

test_file = openf(path_test_file)
Y = []
Y_name = []
for file in test_file:
    im = Image.open(str(file.split('\n')[0])).convert('L')
    array = asarray(im)
    Y.append(array.reshape([-1]))
    Y_name.append(str(file.split('\n')[0]))
    im.close


# In[5]:
def matmul(a,b):
    ans = np.matmul(a,b)
    return ans

def mean(a):
    ans = np.mean(a,axis=0)
    return ans

mean = mean(X)
mean_x = X - mean
v, eigenvalues ,eigenvectors  = svd(mean_x)
eigenvectors= eigenvectors.T
x_coeff = matmul(mean_x,eigenvectors)
eigenvectors = eigenvectors.T
X_new = np.array(x_coeff[:,:32])
eigenvectors = eigenvectors[0:32,:]

def pow(a,b):
    ans = math.pow(a,b)
    return ans

# In[6]:


mean_y = Y - mean
y_coeff = matmul(mean_y , eigenvectors.T)
Y_new = y_coeff


# In[7]:


label_bucket = dict()
for i in range(len(X_new)):
    name = X_name[i] 
    im = X_new[i]
    if name not in label_bucket.keys():
        label_bucket[name] = list()
    label_bucket[name].append(im)


# In[8]:


mean_bucket = dict()
classes = list()
for i in label_bucket.keys():
    lb = label_bucket[i]
    mean_bucket[i] = np.mean(lb,axis=0)

for i in label_bucket.keys():
    classes.append(i)


# In[9]:
def exp(a):
    ans = math.exp(a)
    return ans


variance_bucket = dict()
for i in label_bucket.keys():
    variance_bucket[i] = np.std(label_bucket[i],axis=0,ddof=1)
# variance_bucket


# In[13]:

def sqrt(a):
    ans = math.sqrt(a)
    return ans



for im in Y_new:
    final_prob_arr = []
    for i in label_bucket.keys():
        total_prob = 1
        for j in range(32):
            x = im[j]
            var1 = 1
            mean = mean_bucket[i][j]
            var = variance_bucket[i][j]
            prob = calc_prob(x,mean,var,var1)
            total_prob *= prob
        num = len(label_bucket[i])
        den = len((X_new))
        prob_class = num/den
        final_prob = total_prob*prob_class
        final_prob_arr.append(final_prob)
    max_ind = max(final_prob_arr)
    ind = final_prob_arr.index(max_ind)
    print(classes[ind])

