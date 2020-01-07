import numpy as np
from matplotlib import pyplot as plt
#Input data
x = np.array([[1 , 2, -1],
            [2 , 3, -1],
            [2 , 1, -1],
            [3 , 4, -1],
            [1 , 3, -1],
            [4 , 4, -1],])
y = np.array([-1, 1, -1, 1, -1, 1]).reshape(-1, 1)

xtrain = x[:4, :]
ytrain = y[:4, :]

xtest = x[4:, :]
ytest = y[4:, :]


# In[227]:
# Not Optimised yet
np.random.seed(6)
w = np.random.randn(3)

out = np.dot(x,w)

result = np.dot(xtest,w)

print("\n\nBefore optimising the model")
print('Initialised Weights: ',w)
print('Predicted Outputs (not optimised yet): ', out)
print('Test Predictions (not optimised yet): ',result)

# Plotting
for val, inp in enumerate(x):
    if y[val] == -1:
        plt.scatter(inp[0], inp[1], marker='_', linewidths=5)
    else:
        plt.scatter(inp[0], inp[1], marker='+', linewidths=5)
x1=[w[0],w[1],-w[1],w[0]]
x2=[w[0],w[1],w[1],-w[0]]

x1x2 = np.array([x1,x2])
# print(x1x2)
X,Y,U,V = zip(*x1x2)
ax = plt.gca()
plt.title("Before optimising the model")
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()

# In[227]:

def svmfit(x, y, learning_rate=0.1):
    # Initialization
    np.random.seed(6)
    w = np.random.randn(3)
    l_rate = learning_rate #Hyperparameter
    epochs = 4 #Can be decreased :P
    for e in range(epochs):
        val1 = np.dot(x,w).reshape(-1,1)
        # print((l_rate*(y*x)*(y*val1<1)).shape)
        if ((y*val1)<1).sum()==0: #For early stopping
            print("Halted at :",e)
            break
        print(f"W at {e} is {w}")
        print(f"Output at {e} is {val1}")
        for k in x:
            print(f"| {k[0]:3.3f}x{w[0]:3.3f} + {k[1]:3.3f}x{w[1]:3.3f} + {k[2]:3.3f}x{w[2]:3.3f} |")
        if e==4:
            pass
        w = w + (l_rate*(y*x)*(y*val1<1)).sum(axis=0)
    out = np.dot(x,w)
    return w, out


# In[241]:

w, out = svmfit(x, y, learning_rate=0.01)


# In[242]:
print("\n\nAfter optimising the model")
print('Optimised Weights: ',w)

# In[243]:

print('Predicted Outputs: ', out)


# In[245]:

result = np.dot(xtest,w)
print('Test Data Predictions: ',result)

# In[247]:

for val, inp in enumerate(x):
    if y[val] == -1:
        plt.scatter(inp[0], inp[1], marker='_', linewidths=5)
    else:
        plt.scatter(inp[0], inp[1], marker='+', linewidths=5)


x1=[w[0],w[1],-w[1],w[0]]
x2=[w[0],w[1],w[1],-w[0]]

x1x2 = np.array([x1,x2])
# print(x1x2)
X,Y,U,V = zip(*x1x2)
ax = plt.gca()
plt.title("After optimising the model")
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()
