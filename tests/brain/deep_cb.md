```python
import numpy as np

class Model:
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        h = np.maximum(0, np.dot(X, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out
    
    def backward(self, X, y, output):
        delta_output = output - y
        delta_hidden = np.dot(delta_output, self.W2.T) * (self.W1 > 0)
        h = np.maximum(0, np.dot(X, self.W1) + self.b1)
        self.W2 -= 0.1 * np.dot(h.T, delta_output)
        self.b2 -= 0.1 * np.sum(delta_output, axis=0, keepdims=True)
        self.W1 -= 0.1 * np.dot(X.T, delta_hidden)
        self.b1 -= 0.1 * np.sum(delta_hidden, axis=0, keepdims=True)



def choose_action(model, context, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.choice(model.output_dim)
    else:
        return np.argmax(model.forward(context))

import random

# generate some fake data
contexts = np.random.randn(100, 1)
actions = np.random.randint(3, size=100)
rewards = np.zeros(100)
for i in range(100):
    if actions[i] == 0:
        rewards[i] = np.random.normal(1, 0.1) * contexts[i]
    elif actions[i] == 1:
        rewards[i] = np.random.normal(2, 0.1) * contexts[i]
    else:
        rewards[i] = np.random.normal(0, 0.1) * contexts[i]

# train the model
model = Model(input_dim=1, output_dim=3, hidden_dim=10)
epsilon = 0.1
for t in range(1000):
    idx = random.randint(0, len(contexts) - 1)
    context = contexts[idx]
    action = choose_action(model, context, epsilon)
    reward = rewards[idx]
    y = np.zeros(3)
    y[action] = reward
    output = model.forward(context.reshape(1,-1))
    model.backward(context.reshape(1,-1), y, output)

```


```python
model.W1
```




    array([[-1.49031908e+00, -4.44896190e-01, -1.14228501e-02,
            -1.40589308e+00, -9.64780214e-01, -1.51423217e-01,
            -3.90400883e-01, -3.83275882e-04, -4.76242759e-01,
            -7.79588045e-03]])




```python
context
```




    array([0.50267041])




```python
y,output,context.reshape(1,-1)
```




    (array([0.        , 0.09431802, 0.        ]),
     array([[-0.28618451, -0.03554534, -2.82803425]]),
     array([[-1.55637991]]))




```python
output-y
```




    array([[-0.28618451, -0.12986336, -2.82803425]])




```python
model.forward(contexts[0])
```




    array([[0.91272596, 0.21799937, 0.09747264]])




```python
xx=context.reshape(1,-1)
model.forward(xx)
```




    array([[-0.11498612,  0.0421402 , -1.13627631]])




```python
model.forward(context)
```




    array([[-0.11498612,  0.0421402 , -1.13627631]])




```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.zeros((1,10))
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])




```python
from minirl.brain.deep_cb import DeepCBAgent
from mlopskit import make

model_db = make("cache/feature_store-v1", db_name="deepcb.db")

test=DeepCBAgent(4,3,64,0.1,model_db)

```

    2023-05-28 14:51:00 [info     ] APIs of mlopskit               model_name=feature_store model_version=1 ops_type=cache
    2023-05-28 14:51:01 [info     ] The list of all versions for the current model is [42;1m[1, 2][0m.
    2023-05-28 14:51:01 [warning  ] The version 1 is out of date. You should consider upgrading to version `v2`.
    2023-05-28 14:51:01 [info     ] Usage of mlopskit-cache        Params=[36;1m{'db_type': 'rlite/redis/sfdb/diskcache, default:rlite', 'return_type': 'dblink/dbobj, default: dbobj', 'db_name': 'default: rlite_model.cache'}[0m
    An actor network is created.



```python

for i in range(10):
    xxxx= test.act([1,0,0,0],"test1")
    print(xxxx)
```

    0
    0
    0
    0
    0
    0
    0
    0
    0
    0



```python
test.learn([1,0,0,0],2,1.0,"test1")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 test.learn([1,0,0,0],2,1.0,"test1")


    File ~/miniconda3/lib/python3.8/site-packages/minirl/brain/deep_cb.py:115, in DeepCBAgent.learn(self, context, action, reward, model_id)
        112 y[action] = reward
        113 output = self.model.forward(context.reshape(1, -1))
    --> 115 self.model.backward(context.reshape(1, -1), y, output)
        116 self.model.save_weights(model_id)


    File ~/miniconda3/lib/python3.8/site-packages/minirl/brain/deep_cb.py:39, in Model.backward(self, X, y, output)
         37 W2 -= 0.1 * np.dot(h.T, delta_output)
         38 b2 -= 0.1 * np.sum(delta_output, axis=0, keepdims=True)
    ---> 39 W1 -= 0.1 * np.dot(X.T, delta_hidden)
         40 b1 -= 0.1 * np.sum(delta_hidden, axis=0, keepdims=True)
         42 self.params["W1"] = W1


    File <__array_function__ internals>:180, in dot(*args, **kwargs)


    ValueError: shapes (4,1) and (4,64) not aligned: 1 (dim 1) != 4 (dim 0)



```python
np.array([1,0,0,0]).reshape(1, -1).T
```




    array([[1],
           [0],
           [0],
           [0]])




```python
from minirl.brain.deep_cb import DeepCB
test=DeepCB(4,3)
```


```python
for i in range(10):
    p=test.select_action([1,0,0,0])
    print(p)
```

    2
    2
    2
    2
    2
    2
    2
    2
    2
    2



```python
import numpy as np
x=np.array([1,0,0,0])
a=0
r=1.0
test.update(x,a,r)
```


    ---------------------------------------------------------------------------

    AxisError                                 Traceback (most recent call last)

    Cell In[7], line 5
          3 a=0
          4 r=1.0
    ----> 5 test.update(x,a,r)


    File ~/miniconda3/lib/python3.8/site-packages/minirl/brain/deep_cb.py:150, in DeepCB.update(self, x, a, r)
        147 def update(self, x, a, r):
        148     # compute gradient of loss w.r.t. weights
        149     logits = self.predict(x)
    --> 150     probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        151     probs_for_actions = probs[range(len(a)), a]
        152     dlog = probs.copy()


    File <__array_function__ internals>:180, in sum(*args, **kwargs)


    File ~/miniconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:2296, in sum(a, axis, dtype, out, keepdims, initial, where)
       2293         return out
       2294     return res
    -> 2296 return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
       2297                       initial=initial, where=where)


    File ~/miniconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:86, in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
         83         else:
         84             return reduction(axis=axis, out=out, **passkwargs)
    ---> 86 return ufunc.reduce(obj, axis, dtype, out, **passkwargs)


    AxisError: axis 1 is out of bounds for array of dimension 1



```python
import numpy as np

# generate random data for a simple 3-armed bandit problem
N = 1000 # number of trials
D = 2 # number of context features
K = 3 # number of actions
X = np.random.randn(N, D) # matrix of input contexts
theta = np.array([[1, 0], [0, 2]]) # true coefficient matrix
noise = np.random.randn(N, K) # iid Gaussian noise
Y = np.dot(X, theta) + noise # matrix of rewards

cb = DeepCB(feature_dim=D, action_dim=K, learning_rate=0.01, num_hidden_layers=2, hidden_layer_size=32)

for t in range(N):
    # select action based on current context
    a = cb.select_action(X[t])
    
    # compute loss and update weights based on observed reward
    prob = np.exp(cb.predict(X[t])) / np.sum(np.exp(cb.predict(X[t])))
    loss = -np.log(prob[a])
    cb.update(X[t], a, Y[t][a])

# generate test data
N_test = 100
X_test = np.random.randn(N_test, D)
Y_test = np.dot(X_test, theta) + np.random.randn(N_test, K)

# evaluate performance of learned policy on test set
regret = 0
for t in range(N_test):
    true_rewards = np.dot(X_test[t], theta)
    optimal_action = np.argmax(true_rewards)
    chosen_action = cb.select_action(X_test[t])
    regret += true_rewards[optimal_action] - true_rewards[chosen_action]
print('Average regret on test set: {:.2f}'.format(regret / N_test))

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[8], line 10
          8 theta = np.array([[1, 0], [0, 2]]) # true coefficient matrix
          9 noise = np.random.randn(N, K) # iid Gaussian noise
    ---> 10 Y = np.dot(X, theta) + noise # matrix of rewards
         12 cb = DeepCB(feature_dim=D, action_dim=K, learning_rate=0.01, num_hidden_layers=2, hidden_layer_size=32)
         14 for t in range(N):
         15     # select action based on current context


    ValueError: operands could not be broadcast together with shapes (1000,2) (1000,3) 



```python
import numpy as np

# generate random data for a simple 3-armed bandit problem
N = 1000 # number of trials
D = 2 # number of context features
K = 3 # number of actions
X = np.random.randn(N, D) # matrix of input contexts
theta = np.array([[1, 0], [0, 2]]) # true coefficient matrix
noise = np.random.randn(N) # iid Gaussian noise
Y = np.dot(X, theta)[:, 0:1] + noise.reshape(-1, 1) # matrix of rewards

# define the DeepCB class
class DeepCB():
    def __init__(self, feature_dim, action_dim, learning_rate=0.01, num_hidden_layers=1, hidden_layer_size=64):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        
        # initialize weight matrices for all layers
        self.weights = {}
        self.weights['W1'] = np.random.randn(feature_dim, hidden_layer_size) / np.sqrt(feature_dim)
        self.weights['b1'] = np.zeros(hidden_layer_size)
        for i in range(2, num_hidden_layers + 2):
            self.weights['W{}'.format(i)] = np.random.randn(hidden_layer_size, hidden_layer_size) / np.sqrt(hidden_layer_size)
            self.weights['b{}'.format(i)] = np.zeros(hidden_layer_size)
        self.weights['WO'] = np.random.randn(hidden_layer_size, action_dim) / np.sqrt(hidden_layer_size)
        self.weights['bO'] = np.zeros(action_dim)

    def predict(self, x):
        # forward pass through network
        h = x
        for i in range(1, self.num_hidden_layers + 2):
            h = np.maximum(0, np.dot(h, self.weights['W{}'.format(i)]) + self.weights['b{}'.format(i)])
        output = np.dot(h, self.weights['WO']) + self.weights['bO']
        return output

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    def update(self, x, a, r):
        # compute gradient of loss w.r.t. weights
        logits = self.predict(x)
        logits = np.array(logits).reshape(1,-1)
        probs = self._softmax(logits)#np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probs_for_actions = probs[range(len(a)), a]
        dlog = probs.copy()
        dlog[range(len(a)), a] -= 1
        dlog = dlog / len(r)
        grads = {}
        grads['WO'] = np.dot(self.weights['h'].T, dlog)
        grads['bO'] = np.sum(dlog, axis=0)
        dh = np.dot(dlog, self.weights['WO'].T)
        dh[self.weights['h'] <= 0] = 0
        for i in reversed(range(1, self.num_hidden_layers + 1)):
            grads['W{}'.format(i)] = np.dot(self.weights['H{}'.format(i - 1)].T, dh)
            grads['b{}'.format(i)] = np.sum(dh, axis=0)
            dh = np.dot(dh, self.weights['W{}'.format(i)].T)
            dh[self.weights['H{}'.format(i - 1)] <= 0] = 0
        grads['W1'] = np.dot(x.T, dh)
        grads['b1'] = np.sum(dh, axis=0)

        # update weights using Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for param in self.weights:
            m = np.zeros_like(self.weights[param])
            v = np.zeros_like(self.weights[param])
            t = 0
            for i in range(grads[param].shape[0]):
                g = grads[param][i, :]
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                self.weights[param][i, :] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def select_action(self, x):
        # choose action with highest expected reward based on current context
        logits = self.predict(x)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.argmax(probs)

# create an instance of the DeepCB class with appropriate hyperparameters
cb = DeepCB(feature_dim=D, action_dim=K, learning_rate=0.01, num_hidden_layers=2, hidden_layer_size=32)

# run a loop over the entire dataset, selecting an action for each context and updating the network weights based on the observed reward
for t in range(N):
    # select action based on current context
    a = cb.select_action(X[t])
    
    # compute loss and update weights based on observed reward
    prob = np.exp(cb.predict(X[t])) / np.sum(np.exp(cb.predict(X[t])))
    loss = -np.log(prob[a])
    cb.update(X[t], a, Y[t])

# evaluate performance of learned policy on test set
N_test = 100
X_test = np.random.randn(N_test, D)
true_rewards = np.dot(X_test, theta)[:, 0]
regret = 0
for t in range(N_test):
    optimal_action = np.argmax(true_rewards[t])
    chosen_action = cb.select_action(X_test[t])
    regret += true_rewards[t][optimal_action] - true_rewards[t][chosen_action]
print('Average regret on test set: {:.2f}'.format(regret / N_test))

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[19], line 102
        100     prob = np.exp(cb.predict(X[t])) / np.sum(np.exp(cb.predict(X[t])))
        101     loss = -np.log(prob[a])
    --> 102     cb.update(X[t], a, Y[t])
        104 # evaluate performance of learned policy on test set
        105 N_test = 100


    Cell In[19], line 51, in DeepCB.update(self, x, a, r)
         49 logits = np.array(logits).reshape(1,-1)
         50 probs = self._softmax(logits)#np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    ---> 51 probs_for_actions = probs[range(len(a)), a]
         52 dlog = probs.copy()
         53 dlog[range(len(a)), a] -= 1


    TypeError: object of type 'numpy.int64' has no len()



```python
import numpy as np

class DeepContextualBandit:
    
    def __init__(self, n_actions, n_features, hidden_layers):
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        
        # Initialize the network weights and biases
        self.W1 = np.random.randn(self.n_features, self.hidden_layers[0])
        self.b1 = np.zeros((1, self.hidden_layers[0]))
        
        self.W2 = np.random.randn(self.hidden_layers[0], self.n_actions)
        self.b2 = np.zeros((1, self.n_actions))
        
        self.weights = [self.W1, self.b1, self.W2, self.b2]
        
    def policy(self, context):
        # Compute the output probabilities for each possible action
        h1 = np.dot(context, self.W1) + self.b1
        a1 = np.tanh(h1)
        h2 = np.dot(a1, self.W2) + self.b2
        exp_scores = np.exp(h2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs.squeeze()
    
    def update(self, context, action, reward, lr):
        # Compute the loss and gradients
        probs = self.policy(context)
        loss = -np.log(probs[action]) * reward
        dLdh2 = probs
        dLdh2[0, action] -= 1
        dLdW2 = np.dot(np.transpose(a1), dLdh2)
        dLdb2 = np.sum(dLdh2, axis=0, keepdims=True)
        da1dh1 = 1 - np.square(a1)
        dLda1 = np.dot(dLdh2, np.transpose(self.W2))
        dLdh1 = dLda1 * da1dh1
        dLdW1 = np.dot(np.transpose(context), dLdh1)
        dLdb1 = np.sum(dLdh1, axis=0, keepdims=True)
        
        # Update the weights
        self.W2 -= lr * dLdW2
        self.b2 -= lr * dLdb2
        self.W1 -= lr * dLdW1
        self.b1 -= lr * dLdb1
        self.weights = [self.W1, self.b1, self.W2, self.b2]

import numpy as np

# Define the reward distribution
reward_means = np.array([0.1, 0.5, 0.9])
reward_vars = np.array([0.01, 0.05, 0.1])
n_actions = len(reward_means)
n_features = 10
hidden_layers = [20, 10]

# Initialize the contextual bandit and generate training data
bandit = DeepContextualBandit(n_actions, n_features, hidden_layers)
n_episodes = 1000
for episode in range(n_episodes):
    context = np.random.randn(1, n_features)
    probs = bandit.policy(context)
    action = np.random.choice(n_actions, p=probs)
    reward = np.random.normal(reward_means[action], reward_vars[action])
    bandit.update(context, action, reward, lr=0.01)

# Evaluate the performance of the learned policy
n_trials = 1000
total_reward = 0
for trial in range(n_trials):
    context = np.random.randn(1, n_features)
    probs = bandit.policy(context)
    action = np.argmax(probs)
    reward = np.random.normal(reward_means[action], reward_vars[action])
    total_reward += reward

print("Average reward over {} trials: {}".format(n_trials, total_reward / n_trials))

```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[16], line 67
         65     action = np.random.choice(n_actions, p=probs)
         66     reward = np.random.normal(reward_means[action], reward_vars[action])
    ---> 67     bandit.update(context, action, reward, lr=0.01)
         69 # Evaluate the performance of the learned policy
         70 n_trials = 1000


    Cell In[16], line 34, in DeepContextualBandit.update(self, context, action, reward, lr)
         32 loss = -np.log(probs[action]) * reward
         33 dLdh2 = probs
    ---> 34 dLdh2[0, action] -= 1
         35 dLdW2 = np.dot(np.transpose(a1), dLdh2)
         36 dLdb2 = np.sum(dLdh2, axis=0, keepdims=True)


    IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed



```python
import numpy as np

class DeepContextualBandit:
    
    def __init__(self, n_actions, n_features, hidden_layers):
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        
        # Initialize the network weights and biases
        self.W1 = np.random.randn(self.n_features, self.hidden_layers[0])
        self.b1 = np.zeros((1, self.hidden_layers[0]))
        
        self.W2 = np.random.randn(self.hidden_layers[0], self.n_actions)
        self.b2 = np.zeros((1, self.n_actions))
        
        self.weights = [self.W1, self.b1, self.W2, self.b2]

        # Initialize Adam optimizer variables
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
    def policy(self, context):
        # Compute the output probabilities for each possible action
        h1 = np.dot(context, self.W1) + self.b1
        a1 = np.tanh(h1)
        h2 = np.dot(a1, self.W2) + self.b2
        exp_scores = np.exp(h2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs.squeeze()
    
    def update(self, context, action, reward, lr):
        # Compute the loss and gradients
        probs = self.policy(context)
        loss = -np.log(probs[action]) * reward
        dLdh2 = probs
        dLdh2[action] -= 1
        dLdW2 = np.dot(np.transpose(a1), dLdh2)
        dLdb2 = np.sum(dLdh2, axis=0, keepdims=True)
        da1dh1 = 1 - np.square(a1)
        dLda1 = np.dot(dLdh2, np.transpose(self.W2))
        dLdh1 = dLda1 * da1dh1
        dLdW1 = np.dot(np.transpose(context), dLdh1)
        dLdb1 = np.sum(dLdh1, axis=0, keepdims=True)
        
        # Update the Adam optimizer variables
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, [dLdW1, dLdb1, dLdW2, dLdb2])]
        self.v = [self.beta2 * v + (1 - self.beta2) * np.square(g) for v, g in zip(self.v, [dLdW1, dLdb1, dLdW2, dLdb2])]
        m_hat = [m / (1 - self.beta1) for m in self.m]
        v_hat = [v / (1 - self.beta2) for v in self.v]
        
        # Update the weights using the Adam optimizer
        self.W2 -= lr * m_hat[2] / (np.sqrt(v_hat[2]) + self.epsilon)
        self.b2 -= lr * m_hat[3] / (np.sqrt(v_hat[3]) + self.epsilon)
        self.W1 -= lr * m_hat[0] / (np.sqrt(v_hat[0]) + self.epsilon)
        self.b1 -= lr * m_hat[1] / (np.sqrt(v_hat[1]) + self.epsilon)
        self.weights = [self.W1, self.b1, self.W2, self.b2]



# Define the reward distribution
reward_means = np.array([0.1, 0.5, 0.9])
reward_vars = np.array([0.01, 0.05, 0.1])
n_actions = len(reward_means)
n_features = 10
hidden_layers = [20, 10]

# Initialize the contextual bandit and generate training data
bandit = DeepContextualBandit(n_actions, n_features, hidden_layers)
n_episodes = 1000
for episode in range(n_episodes):
    context = np.random.randn(1, n_features)
    probs = bandit.policy(context)
    action = np.random.choice(n_actions, p=probs)
    reward = np.random.normal(reward_means[action], reward_vars[action])
    bandit.update(context, action, reward, lr=0.01)

# Evaluate the performance of the learned policy
n_trials = 1000
total_reward = 0
for trial in range(n_trials):
    context = np.random.randn(1, n_features)
    probs = bandit.policy(context)
    action = np.argmax(probs)
    reward = np.random.normal(reward_means[action], reward_vars[action])
    total_reward += reward

print("Average reward over {} trials: {}".format(n_trials, total_reward / n_trials))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[24], line 80
         78     action = np.random.choice(n_actions, p=probs)
         79     reward = np.random.normal(reward_means[action], reward_vars[action])
    ---> 80     bandit.update(context, action, reward, lr=0.01)
         82 # Evaluate the performance of the learned policy
         83 n_trials = 1000


    Cell In[24], line 42, in DeepContextualBandit.update(self, context, action, reward, lr)
         40 dLdh2 = probs
         41 dLdh2[action] -= 1
    ---> 42 dLdW2 = np.dot(np.transpose(a1), dLdh2)
         43 dLdb2 = np.sum(dLdh2, axis=0, keepdims=True)
         44 da1dh1 = 1 - np.square(a1)


    NameError: name 'a1' is not defined



```python
n_features = 4
n_actions=3
# Initialize the contextual bandit and generate training data
bandit = DeepContextualBandit(n_actions, n_features, hidden_layers)

bandit.policy([1,0,0,0])
```




    array([0.44397411, 0.45982688, 0.09619901])




```python
context = [1,0,0,0]
action=0
reward=1
bandit.update(context, action, reward, lr=0.01)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[26], line 4
          2 action=0
          3 reward=1
    ----> 4 bandit.update(context, action, reward, lr=0.01)


    Cell In[24], line 42, in DeepContextualBandit.update(self, context, action, reward, lr)
         40 dLdh2 = probs
         41 dLdh2[action] -= 1
    ---> 42 dLdW2 = np.dot(np.transpose(a1), dLdh2)
         43 dLdb2 = np.sum(dLdh2, axis=0, keepdims=True)
         44 da1dh1 = 1 - np.square(a1)


    NameError: name 'a1' is not defined



```python
import numpy as np

# Define the environment
class Environment:
    def __init__(self, num_actions):
        self.num_actions = num_actions
    
    def generate_context(self):
        return np.random.normal(size=(10,))
    
    def get_reward(self, action, context):
        return np.random.normal(loc=action + np.dot(context, np.random.normal(size=(10,))), scale=0.5)

# Define the neural network
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.params = {}
        
        # Initialize weights for each layer
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.params['W' + str(i)] = np.random.randn(dims[i], dims[i+1])
            self.params['b' + str(i)] = np.zeros(dims[i+1])
    
    def forward(self, x):
        
        h = x
        for i in range(len(self.params) // 2 - 1):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            h = np.maximum(0, np.dot(h, W) + b)
        W = self.params['W' + str(len(self.params) // 2 - 1)]
        b = self.params['b' + str(len(self.params) // 2 - 1)]
        y = softmax(np.dot(h, W) + b)
        return y
        
    def backward(self, x, y, action, reward):
        x=np.reshape(x,[1,-1])
        grads = {}
        
        # Calculate derivative of loss w.r.t. output
        dL_dy = np.zeros(y.shape)
        dL_dy[action] = -reward / y[action]
        
        # Backpropagate through network
        h = x

        for i in range(len(self.params) // 2 - 1):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            print(W.T.shape,dL_dy)
            dh = np.dot(dL_dy, W.T)
            dW = np.dot(h.T, (dh * (h > 0)))
            db = np.sum(dh * (h > 0), axis=0)
            grads['W' + str(i)] = dW
            grads['b' + str(i)] = db
            dL_dy = dh * (h > 0)
            h = np.maximum(0, np.dot(h, W.T) + b)
        W = self.params['W' + str(len(self.params) // 2 - 1)]
        b = self.params['b' + str(len(self.params) // 2 - 1)]
        dW = np.dot(h.T, dL_dy)
        db = np.sum(dL_dy, axis=0)
        grads['W' + str(len(self.params) // 2 - 1)] = dW
        grads['b' + str(len(self.params) // 2 - 1)] = db
        
        return grads



# Implement the training loop
env = Environment(num_actions=5)
nn = NeuralNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=5)
num_steps = 10000
lr = 0.01
for i in range(num_steps):
    context = env.generate_context()
    probs = nn.forward(context)
    print(probs)
    action = np.random.choice(env.num_actions, p=probs)
    reward = env.get_reward(action, context)
    grads = nn.backward(context, probs, action, reward)
    
    # Update weights using Adam optimizer
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = i + 1
    for param_name in nn.params:
        m = np.zeros_like(nn.params[param_name])
        v = np.zeros_like(nn.params[param_name])
        m = beta1 * m + (1 - beta1) * grads[param_name]
        v = beta2 * v + (1 - beta2) * (grads[param_name] ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        nn.params[param_name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

# Evaluate the performance
cumulative_reward = 0
optimal_reward = 0
for i in range(num_steps):
    context = env
    probs = nn.forward(context)
    action = np.argmax(probs)
    reward = env.get_reward(action, context)
    cumulative_reward += reward
    optimal_reward += np.max([env.get_reward(a, context) for a in range(env.num_actions)])
regret = optimal_reward - cumulative_reward
print('Regret:', regret)




```

    [1.36999113e-48 2.07537421e-73 4.31057064e-36 1.78743989e-39
     1.00000000e+00]
    (64, 10) [0.         0.         0.         0.         3.81115815]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[212], line 85
         83 action = np.random.choice(env.num_actions, p=probs)
         84 reward = env.get_reward(action, context)
    ---> 85 grads = nn.backward(context, probs, action, reward)
         87 # Update weights using Adam optimizer
         88 beta1 = 0.9


    Cell In[212], line 56, in NeuralNetwork.backward(self, x, y, action, reward)
         54 b = self.params['b' + str(i)]
         55 print(W.T.shape,dL_dy)
    ---> 56 dh = np.dot(dL_dy, W.T)
         57 dW = np.dot(h.T, (dh * (h > 0)))
         58 db = np.sum(dh * (h > 0), axis=0)


    File <__array_function__ internals>:180, in dot(*args, **kwargs)


    ValueError: shapes (5,) and (64,10) not aligned: 5 (dim 0) != 64 (dim 0)



```python
import numpy as np

class DeepContextualBandit:
    def __init__(self, n_context_features, n_actions, n_hidden_units):
        self.theta = np.random.normal(size=(n_context_features, n_hidden_units))
        self.W = np.random.normal(size=(n_hidden_units, n_actions))

    def predict(self, x):
        hidden_layer = np.dot(x, self.theta)
        logits = np.dot(hidden_layer, self.W)
        return softmax(logits)

    def update(self, x, a, r, lr=0.001):
        p = self.predict(x)
        log_prob = np.log(p[a])
        loss = -log_prob * r
        dloss_dlogits = p.copy()
        dloss_dlogits[a] -= 1
        dloss_dW = np.dot(x.T.reshape(-1,1), dloss_dlogits.reshape(1,-1))
        dloss_dtheta = np.dot(dloss_dlogits.reshape(1,-1), self.W.T) * (1 - np.tanh(np.dot(x, self.theta))**2)
        self.theta -= lr * dloss_dtheta
        self.W -= lr * dloss_dW

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Example usage
n_context_features = 10
n_actions = 5
n_hidden_units = 50
bandit = DeepContextualBandit(n_context_features, n_actions, n_hidden_units)
num_iterations=4
epsilon=0.1
for i in range(num_iterations):
    # Sample context
    x = np.random.normal(size=n_context_features)

    # Make decision based on current policy
    if np.random.rand() < epsilon:
        a = np.random.choice(np.arange(n_actions))
    else:
        a = np.argmax(bandit.predict(x))

    # Sample reward
    r = 1.0 #simulate_reward(x, a)

    # Update policy based on observed reward signal
    bandit.update(x, a, r)

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[207], line 49
         46 r = 1.0 #simulate_reward(x, a)
         48 # Update policy based on observed reward signal
    ---> 49 bandit.update(x, a, r)


    Cell In[207], line 22, in DeepContextualBandit.update(self, x, a, r, lr)
         20 dloss_dtheta = np.dot(dloss_dlogits.reshape(1,-1), self.W.T) * (1 - np.tanh(np.dot(x, self.theta))**2)
         21 self.theta -= lr * dloss_dtheta
    ---> 22 self.W -= lr * dloss_dW


    ValueError: operands could not be broadcast together with shapes (50,5) (10,5) (50,5) 



```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GatedLinearNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        h = np.maximum(0, np.dot(X, self.W1) + self.b1)  # ReLU activation
        g = sigmoid(np.dot(h, self.W2) + self.b2)  # Sigmoid gating
        return g * h  # Element-wise multiplication for gating

    def backward(self, X, y, p, lr=0.01):
        # Compute gradients using backpropagation
        dh = (p - y) @ self.W2.T * (self.forward(X) > 0)
        dW2 = self.forward(X).T @ (p - y)
        db2 = np.sum(p - y, axis=0)
        dW1 = X.T @ dh
        db1 = np.sum(dh, axis=0)

        # Update weights and biases using stochastic gradient descent
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


import numpy as np

# Define the GLN architecture
input_size = 10  # number of features in the input context
hidden_size = 20  # number of hidden units in the GLN
output_size = 5  # number of possible actions in the bandit problem
gln = GatedLinearNetwork(input_size, hidden_size, output_size)

# Generate some training data
X_train = np.random.randn(1000, input_size)
y_train = np.random.randint(output_size, size=1000)

# Train the GLN using online learning
for i in range(len(X_train)):
    x = X_train[i]
    y = y_train[i]
    p = gln.forward(x.reshape(1, -1))
    gln.backward(x.reshape(1, -1), y, p)

# Generate some test data
X_test = np.random.randn(100, input_size)

# Make predictions on the test data
y_pred = gln.predict(X_test)

# Evaluate the accuracy of the predictions
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[46], line 54
         52     x = X_train[i]
         53     y = y_train[i]
    ---> 54     p = gln.forward(x.reshape(1, -1))
         55     gln.backward(x.reshape(1, -1), y, p)
         57 # Generate some test data


    Cell In[46], line 18, in GatedLinearNetwork.forward(self, X)
         16 h = np.maximum(0, np.dot(X, self.W1) + self.b1)  # ReLU activation
         17 g = sigmoid(np.dot(h, self.W2) + self.b2)  # Sigmoid gating
    ---> 18 return g * h


    ValueError: operands could not be broadcast together with shapes (1,5) (1,20) 



```python
import numpy as np

# Define the neural network architecture
input_size = 10
hidden_size = 20
output_size = 5

W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)

W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Define the forward pass function
def forward(x):
    h = np.matmul(x, W1) + b1
    h_relu = np.maximum(h, 0)
    y_pred = np.matmul(h_relu, W2) + b2
    return y_pred

# Define the reward function
def reward(state, action):
    # Return a random reward for this example
    return np.random.normal()

# Define the update function using the Adam optimizer
def update(params, grad, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = params.get("m", 0)
    v = params.get("v", 0)
    t = params.get("t", 0)

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * np.square(grad)
    t += 1

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    params["m"] = m
    params["v"] = v
    params["t"] = t
    
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)

# Train the agent
n_episodes = 1000
n_steps_per_episode = 100

params = {"m": 0, "v": 0, "t": 0}
for i in range(n_episodes):
    state = np.random.randn(input_size)
    for j in range(n_steps_per_episode):
        action_values = forward(state)
        action = np.argmax(action_values)

        r = reward(state, action)

        # Compute the loss and gradients
        q_values = forward(state)
        q_target = q_values.copy()
        q_target[action] = r + np.max(q_values)

        loss = np.square(q_target - q_values).sum()
        grad_q_values = 2 * (q_values - q_target)

        grad_W2 = np.outer(h_relu, grad_q_values)
        grad_b2 = grad_q_values
        grad_h_relu = np.matmul(grad_q_values, W2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_W1 = np.outer(x, grad_h)
        grad_b1 = grad_h

        # Update the weights using the Adam optimizer
        grads = {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}
        for param_name, grad in grads.items():
            update(params[param_name], grad)
        
        state = np.random.randn(input_size)

# Evaluate the agent
n_test_episodes = 100
total_reward = 0
for i in range(n_test_episodes):
    state = np.random.randn(input_size)
    for j in range(n_steps_per_episode):
        action_values = forward(state)
        action = np.argmax(action_values)

        r = reward(state, action)
        total_reward += r

        state = np.random.randn(input_size)

avg_reward = total_reward / (n_test_episodes * n_steps_per_episode)
print("Average reward:", avg_reward)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[47], line 66
         63 loss = np.square(q_target - q_values).sum()
         64 grad_q_values = 2 * (q_values - q_target)
    ---> 66 grad_W2 = np.outer(h_relu, grad_q_values)
         67 grad_b2 = grad_q_values
         68 grad_h_relu = np.matmul(grad_q_values, W2.T)


    NameError: name 'h_relu' is not defined



```python
import numpy as np

def get_reward(features, action):
    # Generate a reward for the chosen action based on the true weights and feature values
    return np.dot(features, true_weights[:, action])

class DeepContextualBandit:
    def __init__(self, n_features, n_actions, learning_rate=0.01):
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Initialize the model weights randomly
        self.weights = np.random.randn(n_features, n_actions)
        
        # Initialize the Adam optimizer parameters
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0
        
    def predict(self, features):
        # Predict the action probabilities for the given features
        logits = np.dot(features, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs
    
    def update(self, features, action, reward):
        # Compute the TD-error for the chosen action
        probs = self.predict(features)
        td_error = reward - np.dot(features, self.weights[:, action])
        
        # Compute the gradients with respect to the weights
        grad = np.outer(features, -td_error * probs)
        
        # Update the Adam optimizer parameters
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        
        # Compute the bias-corrected estimates of the first and second moments
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)
        
        # Update the model weights using the Adam optimizer
        eps = 1e-8
        alpha = self.learning_rate * np.sqrt(1 - 0.999**self.t) / (1 - 0.9**self.t)
        self.weights -= alpha * m_hat / (np.sqrt(v_hat) + eps)

# Define the number of features and actions
n_features = 5
n_actions = 3

# Initialize the true weights randomly
true_weights = np.random.randn(n_features, n_actions)

# Create an instance of the DeepContextualBandit class
model = DeepContextualBandit(n_features, n_actions)

# Generate a set of features for testing
features = np.random.randn(n_features)

# Choose an action based on the predicted probabilities
probs = model.predict(features)
action = np.random.choice(n_actions, p=probs)

# Generate a reward for the chosen action based on the true weights and feature values
reward = get_reward(features, action)

# Update the model weights using the Adam optimizer and TD-error loss
model.update(features, action, reward)

```


```python

for i in range(10):
    probs = model.predict(features)
    action = np.random.choice(n_actions, p=probs)
    print(action)
```

    2
    2
    2
    2
    2
    2
    2
    2
    2
    2



```python
reward = 10
_action = 0
for i in range(190):
    model.update(features, _action, reward)

```


```python
features
```




    array([ 2.0538958 , -0.45592182, -0.11156368,  2.26930846,  1.83955083])




```python
model.weights
```




    array([[-1.08419051, -0.01487196, -0.08447032],
           [-0.04604203,  0.72478887, -1.55600925],
           [-0.56125922, -0.2741829 , -0.07345757],
           [-0.11980817, -0.21165303,  0.64059228],
           [ 0.28152997,  0.0765191 ,  0.46429352]])




```python
import numpy as np

class DeepContextualBandit:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        ob_n = input_size
        H = hidden_size
        ac_n=output_size
        self.W1 = (-1 + 2 * np.random.rand(ob_n, H)) / np.sqrt(ob_n)
        self.b1 = np.zeros(H)
        self.W2= (-1 + 2 * np.random.rand(H, ac_n)) / np.sqrt(H)
        self.b2 = np.zeros(ac_n)


        # self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        # self.b1 = np.zeros((hidden_size, 1))
        # self.W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        # self.b2 = np.zeros((output_size, 1))

    def predict(self, x):
        # Forward pass through the network
        #z1 = np.dot(self.W1, x) + self.b1
        #a1 = np.maximum(z1, 0)
        #z2 = np.dot(self.W2, a1) + self.b2

        # forward computations
        W1,b1=self.W1,self.b1
        W2,b2 = self.W2,self.b2
    
        affine1 = x.dot(W1) + b1
        relu1 = np.maximum(0, affine1)
        affine2 = relu1.dot(W2) + b2

        logits = affine2  # layer right before softmax (i also call this h)
        # pass through a softmax to get probabilities
        probs = self._softmax(logits)


        return probs

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    def update(self, x, y):
        
        # Compute gradients using backpropagation
        #z1 = np.dot(self.W1, x) + self.b1
        z1 = x.dot(self.W1)+self.b1
        a1 = np.maximum(z1, 0)
        #z2 = np.dot(self.W2, a1) + self.b2
        z2=a1.dot(self.W2)+self.b2
        softmax_out = np.exp(z2) / np.sum(np.exp(z2), axis=0)

        dsoftmax = softmax_out
        dsoftmax[y] -= 1

        dW2 = np.dot(dsoftmax, a1.T)
        db2 = np.reshape(np.sum(dsoftmax, axis=1), (self.output_size, 1))

        da1 = np.dot(self.W2.T, dsoftmax)
        dz1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(dz1, x.T)
        db1 = np.reshape(np.sum(dz1, axis=1), (self.hidden_size, 1))

        # Update weights using Adam optimizer
        self.W1, self.b1, self.W2, self.b2 = adam(self.W1, self.b1, self.W2, self.b2, dW1, db1, dW2, db2, self.learning_rate)

def adam(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Initialize moments to zero
    W1_m = np.zeros_like(W1)
    b1_m = np.zeros_like(b1)
    W2_m = np.zeros_like(W2)
    b2_m = np.zeros_like(b2)

    # Initialize variances to zero
    W1_v = np.zeros_like(W1)
    b1_v = np.zeros_like(b1)
    W2_v = np.zeros_like(W2)
    b2_v = np.zeros_like(b2)

    t = 0 # initialize timestep

    # Update biased moment estimates
    W1_m = beta1 * W1_m + (1 - beta1) * dW1
    b1_m = beta1 * b1_m + (1 - beta1) * db1
    W2_m = beta1 * W2_m + (1 - beta1) * dW2
    b2_m = beta1 * b2_m + (1 - beta1) * db2

    # Update biased variance estimates
    W1_v = beta2 * W1_v + (1 - beta2) * np.square(dW1)
    b1_v = beta2 * b1_v + (1 - beta2) * np.square(db1)
    W2_v = beta2 * W2_v + (1 - beta2) * np.square(dW2)
    b2_v = beta2 * b2_v + (1 - beta2) * np.square(db2)

    # Compute bias-corrected moment and variance estimates
    t += 1
    W1_m_corr = W1_m / (1 - beta1 ** t)
    b1_m_corr = b1_m / (1 - beta1 ** t)
    W2_m_corr = W2_m / (1 - beta1 ** t)
    b2_m_corr = b2_m / (1 - beta1 ** t)

    W1_v_corr = W1_v / (1 - beta2 ** t)
    b1_v_corr = b1_v / (1 - beta2 ** t)
    W2_v_corr = W2_v / (1 - beta2 ** t)
    b2_v_corr = b2_v / (1 - beta2 ** t)

    # Update weights
    W1 -= learning_rate * W1_m_corr / (np.sqrt(W1_v_corr) + eps)
    b1 -= learning_rate * b1_m_corr / (np.sqrt(b1_v_corr) + eps)
    W2 -= learning_rate * W2_m_corr / (np.sqrt(W2_v_corr) + eps)
    b2 -= learning_rate * b2_m_corr / (np.sqrt(b2_v_corr) + eps)

    return W1, b1, W2, b2


import numpy as np

class Bandit:
    def __init__(self, num_actions, context_dim):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.theta = np.random.randn(context_dim, num_actions)

    def get_context(self):
        # Generate a random context vector
        return np.random.randn(self.context_dim)

    def get_reward(self, context, action):
        # Compute the reward for the given context and action
        return np.dot(context, self.theta[:,action]) + np.random.randn()


import numpy as np

# Create the contextual bandit environment
num_actions = 5
context_dim = 10
bandit = Bandit(num_actions, context_dim)

# Create the deep contextual bandit model with hidden layer size 50
model = DeepContextualBandit(context_dim, 50, num_actions, 0.001)

# Train the model for 10000 steps
for i in range(10000):
    # Get a random context and choose an action based on the current policy
    context = bandit.get_context()
    #action = model.predict(np.reshape(context, (context_dim,1)))
    obs = np.reshape(context,[1,-1])
    action =model.predict(obs)
    action = np.argmax(action)

    # Take the chosen action and observe the reward
    reward = bandit.get_reward(context, action)

    # Update the model weights using Adam optimization
    #x = np.reshape(context, (context_dim,1))
    x = np.reshape(context, [1,-1])
    y = np.array(action)
    model.update(x, y)

# Test the model on 1000 new contexts
total_reward = 0.0
for i in range(1000):
    # Get a new context and choose an action based on the learned policy
    context = bandit.get_context()
    #action = model.predict(np.reshape(context, (context_dim,1)))
    action = model.predict(context)
    action = np.argmax(action)

    # Take the chosen action and observe the reward
    reward = bandit.get_reward(context, action)
    total_reward += reward

# Print the average reward across all test contexts
print("Average reward:", total_reward / 1000.0)


```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[88], line 172
        170     x = np.reshape(context, [1,-1])
        171     y = np.array(action)
    --> 172     model.update(x, y)
        174 # Test the model on 1000 new contexts
        175 total_reward = 0.0


    Cell In[88], line 64, in DeepContextualBandit.update(self, x, y)
         61 softmax_out = np.exp(z2) / np.sum(np.exp(z2), axis=0)
         63 dsoftmax = softmax_out
    ---> 64 dsoftmax[y] -= 1
         66 dW2 = np.dot(dsoftmax, a1.T)
         67 db2 = np.reshape(np.sum(dsoftmax, axis=1), (self.output_size, 1))


    IndexError: index 3 is out of bounds for axis 0 with size 1



```python
context
np.reshape(context, (context_dim,1))
```




    array([[ 0.15354161],
           [-0.68235045],
           [-1.48879375],
           [-0.55490513],
           [-0.83321522],
           [-1.24108735],
           [ 1.93751965],
           [-0.17739913],
           [ 0.14284871],
           [-0.51570859]])




```python
obs=context
obs = np.reshape(obs, [1, -1])
obs
```




    array([[-2.10204149,  0.85236363,  0.98037945, -0.86044372,  0.05011679,
             0.5403553 ,  0.75912338, -0.521934  ,  2.06914535,  0.3697579 ]])




```python
import numpy as np
from numpy.random import choice


def relu(x):
    return np.maximum(0,x)



class DeepContextualBandit():
    def __init__(self, num_actions, num_features, hidden_size=64, learning_rate=0.01):
        self.num_actions = num_actions
        self.num_features = num_features
        self.lr = learning_rate
        
        # Initialize weights for each layer
        self.weights1 = np.random.normal(size=(num_features, hidden_size))
        self.weights2 = np.random.normal(size=(hidden_size, num_actions))
        
        # Initialize biases for each layer
        self.biases1 = np.zeros(hidden_size)
        self.biases2 = np.zeros(num_actions)
        
        # Initialize optimizer: Adam
        self.m1 = np.zeros_like(self.weights1)
        self.v1 = np.zeros_like(self.weights1)
        self.m2 = np.zeros_like(self.weights2)
        self.v2 = np.zeros_like(self.weights2)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.epsilon = self.eps

    def predict(self, context):
        context = np.array(context).reshape(1,-1)
        hidden = relu(np.dot(context, self.weights1) + self.biases1)
        output = softmax(np.dot(hidden, self.weights2) + self.biases2)
        action = choice(self.num_actions, p=output[0])
        return action

    def update2(self, context, action, reward):
        context = np.array(context).reshape(1,-1)
        
        target = np.zeros(self.num_actions)
        target[action] = 1.0
        
        # Forward pass
        hidden = relu(np.dot(context, self.weights1) + self.biases1)
        output = softmax(np.dot(hidden, self.weights2) + self.biases2)
        
        # Backward pass
        d_output = output - target
        d_hidden = np.dot(d_output, self.weights2.T) * (hidden > 0)
        
        # Update weights and biases using Adam optimizer
        grad2 = np.dot(hidden.T, d_output)
        grad1 = np.dot(context.T, d_hidden)
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * grad1
        self.v1 = self.beta2 * self.v1 + (1 - self.beta2) * (grad1 ** 2)
        self.m2 = self.beta1 * self.m2 + (1 - self.beta1) * grad2
        self.v2 = self.beta2 * self.v2 + (1 - self.beta2) * (grad2 ** 2)
        m_hat1 = self.m1 / (1 - self.beta1)
        v_hat1 = self.v1 / (1 - self.beta2)
        m_hat2 = self.m2 / (1 - self.beta1)
        v_hat2 = self.v2 / (1 - self.beta2)
        self.weights1 += self.learning_rate * m_hat1 / (np.sqrt(v_hat1) + self.eps)
        self.weights2 += self.learning_rate * m_hat2 / (np.sqrt(v_hat2) + self.eps)
        self.biases1 += self.learning_rate * np.mean(d_hidden, axis=0)
        self.biases2 += self.learning_rate * np.mean(d_output, axis=0)

    def update(self, context, action, reward):
        context = np.array(context).reshape(1,-1)
        # Forward pass
        hidden = np.maximum(0, np.dot(context, self.weights1) + self.biases1)
        output = np.dot(hidden, self.weights2) + self.biases2

        # Compute loss and gradients
        y = np.zeros_like(output)
        y[action] = 1
        loss = (reward - output[action])**2
        d_output = -(y - output)
        d_hidden = np.dot(d_output, self.weights2.T) * (hidden > 0)

        # Update weights and biases using Adam optimizer
        grad2 = np.dot(hidden.T, d_output)
        grad1 = np.dot(context.T, d_hidden)
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * grad1
        self.m2 = self.beta1 * self.m2 + (1 - self.beta1) * grad2
        self.v1 = self.beta2 * self.v1 + (1 - self.beta2) * grad1**2
        self.v2 = self.beta2 * self.v2 + (1 - self.beta2) * grad2**2
        self.weights1 -= self.lr * self.m1 / (np.sqrt(self.v1) + self.epsilon)
        self.biases1 -= self.lr * np.mean(d_hidden, axis=0)
        self.weights2 -= self.lr * self.m2 / (np.sqrt(self.v2) + self.epsilon)
        self.biases2 -= self.lr * np.mean(d_output, axis=0)

        return loss

# Initialize deep contextual bandit model

num_features =4
num_actions = 3
model = DeepContextualBandit(num_actions, num_features, hidden_size=64, learning_rate=0.01)
```


```python
for i in range(10):
    a=model.predict([1,0,0,0])
    print(a)
```

    0
    2
    1
    1
    0
    2
    2
    0
    0
    2



```python
for i in range(10000):
    reward=1.0
    context=[1,0,0,0]
    action = 0
    model.update(context,action,reward)
```
