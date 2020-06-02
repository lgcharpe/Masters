# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

# %% [markdown]
# ## Install Tensorflow

# %%
import tensorflow as tf
import numpy as np
import copy
import tqdm
import IProgress

# %% [markdown]
# ## Load MNIST Dataset from the tensorflow datasets

# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Converting interger values to floats (0 to 1)

# %% [markdown]
# ## Building the NN model (in this case a simple ANN)

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %% [markdown]
# ## Setting up the optimizer and loss

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %% [markdown]
# ## Training the model

# %%
model.fit(x_train, y_train, epochs=10, batch_size=256)

# %% [markdown]
# ## Evaluating the model

# %%
loss, acc = model.evaluate(x_test, y_test, verbose=2)

# %% [markdown]
# ## Removing random number of nodes

# %%
n = 1
old = model.get_weights()


# %%
def remove_nodes(acc, loss, weights, n, to_test, x_train, y_train, v=0):
    check = 0
    new_loss = loss
    new_acc = acc
    best_score = 1e20
    best_model = copy.deepcopy(weights)
    while check < to_test:   
        new = copy.deepcopy(weights)
        to_drop = np.random.choice(len(new[1]), n, replace=False)
        for i in to_drop:
            new[0][:,i] = 0
            new[1][i] = 0
            new[2][i,:] = 0
        model.set_weights(new)
        new_loss, new_acc = model.evaluate(x_train, y_train, verbose=v)
        score = ((new_loss / loss) - 1) + ((new_acc / acc) - 1)
        if best_score > score:
            best_score = score
            best_model = copy.deepcopy(new)
            nodes_removed = to_drop.copy()
        check = check + 1
    return best_model, best_score, nodes_removed


# %%
check = 0
new_loss = loss
new_acc = acc
best_score = 0
best_model = copy.deepcopy(old)

for i in tqdm.trange(len(new[1])):   
    new = copy.deepcopy(old)
    #for i in range(len(old)):
    #    new[i] = old[i].copy()
    to_drop = np.random.choice(len(new[1]), n, replace=False)
    # for i in to_drop:
    new[0][:,i] = 0
    new[1][i] = 0
    new[2][i,:] = 0
    model.set_weights(new)
    new_loss, new_acc = model.evaluate(x_test, y_test, verbose=0)
    score = (1 - (new_loss / loss)) + ((new_acc / acc) - 1)
    score1 = (1 - (new_loss / loss))
    score2 = ((new_acc / acc) - 1)
    print(score1, score2)
    if best_score < score1 or best_score < score2:
        best_score = score1
        best_model = copy.deepcopy(new)
    if best_score < score2:
        best_score = score2
        best_model = copy.deepcopy(new)
    check = check + 1


# %%
best_model[1]


# %%
model.set_weights(best_model)
loss, acc = model.evaluate(x_test, y_test, verbose=2)


# %%
old[0][130]


# %%
best_score = 0
best_model = copy.deepcopy(old)
to_test = 25
for i in range(1, 65):
    temp_model, temp_score = remove_nodes(acc, loss, old, i, to_test)
    if temp_score < best_score:
        best_model = temp_model
        best_score = temp_score
        print("Found new best model")

# %% [markdown]
# ## Creating new restricted model

# %%
old = model.get_weights()


# %%
n = 2
best_weights, _, nodes_removed = remove_nodes(acc, loss, old, n, 50)

new_weights = [np.zeros((best_weights[0].shape[0], best_weights[0].shape[1] - n)), np.zeros((best_weights[1].shape[0] - n)), np.zeros((best_weights[2].shape[0] - n, best_weights[2].shape[1])), best_weights[3]]

j = 0
for i in range(len(best_weights[1])):
    if i not in nodes_removed:
        new_weights[0][:, j] = best_weights[0][:, i]
        new_weights[1][j] = best_weights[1][i]
        new_weights[2][j, :] = best_weights[2][i, :]
        j = j + 1
    
new_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128 - n, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.set_weights(new_weights)


# %%
def shrink_model(model, shrinkage_factor, x_train, y_train, size, to_test, v=0):
    
    n = shrinkage_factor
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    old = model.get_weights()
    best_weights, _, nodes_removed = remove_nodes(acc, loss, old, n, to_test, x_train, y_train, v)

    new_weights = [np.zeros((best_weights[0].shape[0], best_weights[0].shape[1] - n)), np.zeros((best_weights[1].shape[0] - n)), np.zeros((best_weights[2].shape[0] - n, best_weights[2].shape[1])), best_weights[3]]

    j = 0
    for i in range(len(best_weights[1])):
        if i not in nodes_removed:
            new_weights[0][:, j] = best_weights[0][:, i]
            new_weights[1][j] = best_weights[1][i]
            new_weights[2][j, :] = best_weights[2][i, :]
            j = j + 1

    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(size - n, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.set_weights(new_weights)
    return new_model, size-n


# %%
new_model.evaluate(x_test, y_test, verbose=2)


# %%
new_model.fit(x_train, y_train, epochs=3)


# %%
new_model.evaluate(x_test, y_test, verbose=2)


# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
size = 128
to_test = 25
for _ in range(4):
    model.fit(x_train, y_train, epochs=1)
    model, size = shrink_model(model, 8, x_train, y_train, size, to_test)
    print(len(model.get_weights()[1]))
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test, verbose=2)


# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
rep = 6

best_models = []
sizes = []
scores = []
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
start_weights = copy.deepcopy(model.get_weights())
model.fit(x_train, y_train, epochs=7)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("#############################")
best_models += [model]
scores += [(loss, acc)]
sizes +=[128]
for i in range(1, 16):
    print(f"Starting to shrinking the model by {i}")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.set_weights(start_weights)
    size = 128
    to_test = 25
    for _ in range(rep):
        model.fit(x_train, y_train, epochs=1)
        model, size = shrink_model(model, i, x_train, y_train, size, to_test)
    model.fit(x_train, y_train, epochs=1)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("#############################")
    best_models += [model]
    scores += [(loss, acc)]
    sizes +=[128-(i*rep)]


# %%
print(scores)
print(sizes)


# %%
scores_plain = [scores[0]]
for i in range(1, len(scores)):
    print(f"Starting plain train of Dense size {sizes[i]}")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(sizes[i], activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=7)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    scores_plain += [(loss, acc)]
    print("###############################")


# %%
print(scores)
print(scores_plain)


# %%
for i in range(len(scores)):
    print("Loss change:", (scores_plain[i][0] - scores[i][0])/scores_plain[i][0] *100, "--- Acc change:", -(scores_plain[i][1] - scores[i][1]) / scores_plain[i][1] * 100)


# %%
to_remove_list = np.arange(1, 65)
num_rep = 100
loss_diff = np.zeros(num_rep)
acc_diff = np.zeros(num_rep)
loss_change = np.zeros(num_rep)
acc_change = np.zeros(num_rep)
nodes_removed_list = []
num_nodes_removed = np.zeros(num_rep)


# %%
print(to_remove_list)


# %%
for i in range(num_rep):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    
    n = np.random.choice(to_remove_list, 1)
    
    best_weights, _, nodes_removed = remove_nodes(acc, loss, model.get_weights(), n, 1, x_train, y_train, 0)
    
    model.set_weights(best_weights)
    print(n)
    
    loss_new, acc_new = model.evaluate(x_test, y_test, verbose=2)
    
    loss_diff[i] = loss - loss_new
    acc_diff[i] = acc_new - acc
    loss_change[i] = loss_diff[i] / loss * 100
    acc_change[i] = acc_diff[i] / acc * 100
    num_nodes_removed[i] = n
    nodes_removed_list += [nodes_removed]
    


# %%
for i in range(1, 65):
    print(f"{i} nodes removed")
    print("Loss changes:",loss_change[num_nodes_removed == i])
    print("Accuracy changes:",acc_change[num_nodes_removed == i])
    print("#########################")


# %%
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
end_not_reached = True
improved = False
size = 128
tol = -1e-30
current_pos = 0
best_pos = -1
best_change = tol
original = model.get_weights()
bas = [acc]
bls = [loss]
best_weights = model.get_weights()
nodes_removed = []
best_acc = 0
best_loss = 1e20
ol = loss
oa = acc
num_removed = 0
while end_not_reached or improved:
    if not(end_not_reached):
        end_not_reached = True
        improved = False
        current_pos = 0
        size -= 1
        nodes_removed += [best_pos]
        best_weights[0][:,best_pos] = 0
        best_weights[1][best_pos] = 0
        best_weights[2][best_pos,:] = 0
        best_pos = -1
        tol -= best_change
        ol = best_loss
        oa = best_acc
        bas += [best_acc]
        bls += [best_loss]
        print("Improvement has occured!! Accuracy:", best_acc, "--- Loss:", best_loss, '--- Change:', best_change, '--- New tol:', tol)
        best_change = tol
        num_removed += 1
    if current_pos in nodes_removed:
        current_pos += 1
        if current_pos - num_removed >= size:
            end_not_reached = False
        continue
    w = copy.deepcopy(best_weights)
    w[0][:,current_pos] = 0
    w[1][current_pos] = 0
    w[2][current_pos,:] = 0
    model.set_weights(w)
    nl, na = model.evaluate(x_test, y_test, verbose=0)
    if 0.8*(na - oa) + 0.2*(ol - nl) >= best_change:
        best_change = 0.8*(na - oa) + 0.2*(ol - nl)
        print(best_change)
        best_pos = current_pos
        improved = True
        best_acc = na
        best_loss = nl
        print("Found something better")
    current_pos += 1
    if current_pos - num_removed >= size:
        end_not_reached = False
    if current_pos%20 == 0:
        print("Did 20 iterations")

model.set_weights(best_weights)
loss2, acc2 = model.evaluate(x_test, y_test, verbose=2)


# %%
print(loss - loss2)
print(acc2 - acc)
print((loss - loss2)/loss * 100)
print((acc2 - acc)/acc * 100)
print(num_removed)
print(best_weights[1])


# %%
or_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
or_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
or_model.set_weights(original)
print(or_model.get_weights()[1])


# %%
model.evaluate(x_train, y_train, verbose=2)


# %%
or_model.evaluate(x_train, y_train, verbose=2)


# %%
model.evaluate(x_test, y_test, verbose=2)


# %%
or_model.evaluate(x_test, y_test, verbose=2)


# %%
red_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
red_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
red_model.fit(x_train, y_train, epochs=100, verbose=1)
red_model.evaluate(x_test, y_test, verbose=2)


# %%
red_model.fit(x_train, y_train, epochs=15, verbose=1)
red_model.evaluate(x_test, y_test, verbose=2)


# %%
get_ipython().system('pip install -q pyyaml h5py  ')
# Required to save models in HDF5 format


# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)


# %%
model.save_weights('./reduced/fashion_mnist_128_12')


# %%
or_model.save_weights('./original/fashion_mnist_128_12')


# %%
new_weights = [np.zeros((best_weights[0].shape[0], best_weights[0].shape[1] - num_removed)), np.zeros((best_weights[1].shape[0] - num_removed)), np.zeros((best_weights[2].shape[0] - num_removed, best_weights[2].shape[1])), best_weights[3]]

j = 0
for i in range(len(best_weights[1])):
    if i not in nodes_removed:
        new_weights[0][:, j] = best_weights[0][:, i]
        new_weights[1][j] = best_weights[1][i]
        new_weights[2][j, :] = best_weights[2][i, :]
        j = j + 1


# %%
red_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
red_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
red_model.set_weights(new_weights)
red_model.save_weights('./full_reduced/fashion_mnist_128_12')
red_model.evaluate(x_test, y_test, verbose=2)


# %%
red_model.set_weights(new_weights)
red_model.fit(x_train, y_train, epochs=15, verbose=1, batch_size=4096)
red_model.evaluate(x_test, y_test, verbose=2, batch_size=256)


# %%
new_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.set_weights(best_weights)
new_model.evaluate(x_test, y_test, verbose=2)
new_model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=2048)
new_model.evaluate(x_test, y_test, verbose=2)


# %%
loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
end_not_reached = True
improved = False
size = 128
tol = -1e-30
current_pos = 0
best_pos = -1
best_change = tol
original2 = new_model.get_weights()
bas2 = [acc]
bls2 = [loss]
best_weights2 = new_model.get_weights()
nodes_removed2 = []
best_acc = 0
best_loss = 1e20
ol = loss
oa = acc
num_removed2 = 0
while end_not_reached or improved:
    if not(end_not_reached):
        end_not_reached = True
        improved = False
        current_pos = 0
        size -= 1
        nodes_removed2 += [best_pos]
        best_weights2[0][:,best_pos] = 0
        best_weights2[1][best_pos] = 0
        best_weights2[2][best_pos,:] = 0
        best_pos = -1
        tol -= best_change
        ol = best_loss
        oa = best_acc
        bas2 += [best_acc]
        bls2 += [best_loss]
        print("Improvement has occured!! Accuracy:", best_acc, "--- Loss:", best_loss, '--- Change:', best_change, '--- New tol:', tol)
        best_change = tol
        num_removed2 += 1
    if current_pos in nodes_removed2:
        current_pos += 1
        if current_pos - num_removed2 >= size:
            end_not_reached = False
        continue
    w = copy.deepcopy(best_weights2)
    w[0][:,current_pos] = 0
    w[1][current_pos] = 0
    w[2][current_pos,:] = 0
    new_model.set_weights(w)
    nl, na = new_model.evaluate(x_test, y_test, verbose=0)
    print(f"Node {current_pos}:", 0.8*(na - oa) + 0.2*(ol - nl))
    if 0.8*(na - oa) + 0.2*(ol - nl) > best_change:
        best_change = 0.8*(na - oa) + 0.2*(ol - nl)
        print(best_change)
        best_pos = current_pos
        improved = True
        best_acc = na
        best_loss = nl
        print("Found something better")
    current_pos += 1
    if current_pos - num_removed2 >= size:
        end_not_reached = False
    if current_pos%20 == 0:
        print("Did 20 iterations")

new_model.set_weights(best_weights2)
loss2, acc2 = new_model.evaluate(x_test, y_test, verbose=2)


# %%
tester_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
tester_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# %%
l, a = or_model.evaluate(x_test, y_test, verbose=2)
or_weights = or_model.get_weights()
size = 128
for i in range(128):
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.6*(na - a) + 0.4*(l - nl))


# %%
num_test = 20
num_zeros = np.zeros(num_test)
num_worse = np.zeros(num_test)
num_important = np.zeros(num_test)
losses = np.zeros(num_test)
accs = np.zeros(num_test)
zero_nodes = []
worsening_nodes = []
important_nodes = []
tol = -1e-4
for j in range(num_test):
    blank_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    blank_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    blank_model.fit(x_train, y_train, epochs=10)
    l, a = blank_model.evaluate(x_test, y_test, verbose=2)
    losses[j] = l
    accs[j] = a
    z = []
    wr = []
    imp = []
    for i in range(128):
        w = blank_model.get_weights()
        w[0][:,i] = 0
        w[1][i] = 0
        w[2][i,:] = 0
        tester_model.set_weights(w)
        nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
        change = 0.8*(na - a) + 0.2*(l - nl)
        if change <= 0 and change >= tol:
            num_zeros[j] += 1
            z += [i]
        elif change > 0:
            num_worse[j] += 1
            wr += [i]
        else:
            num_important[j] += 1
            imp += [i]
    zero_nodes += [z]
    worsening_nodes += [wr]
    important_nodes += [imp]


# %%
print(num_zeros)
print(num_worse)
print(num_important)

# %% [markdown]
# # Trying to reduce overfitting through node removal

# %%
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200)


# %%
model.evaluate(x_test, y_test, verbose=2)


# %%
l, a = model.evaluate(x_test, y_test, verbose=2)
or_weights = model.get_weights()
size = 128
for i in range(128):
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.*(na - a) + 1.0*(l - nl))


# %%
loss, acc = model.evaluate(x_test, y_test, verbose=2)
end_not_reached = True
improved = False
size = 128
tol = -1e-30
current_pos = 0
best_pos = -1
best_change = tol
original2 = model.get_weights()
bas2 = [acc]
bls2 = [loss]
best_weights2 = model.get_weights()
nodes_removed2 = []
best_acc = 0
best_loss = 1e20
ol = loss
oa = acc
num_removed2 = 0
while end_not_reached or improved:
    if not(end_not_reached):
        end_not_reached = True
        improved = False
        current_pos = 0
        size -= 1
        nodes_removed2 += [best_pos]
        best_weights2[0][:,best_pos] = 0
        best_weights2[1][best_pos] = 0
        best_weights2[2][best_pos,:] = 0
        best_pos = -1
        #tol -= best_change
        ol = best_loss
        oa = best_acc
        bas2 += [best_acc]
        bls2 += [best_loss]
        print("Improvement has occured!! Accuracy:", best_acc, "--- Loss:", best_loss, '--- Change:', best_change, '--- New tol:', tol)
        best_change = tol
        num_removed2 += 1
    if current_pos in nodes_removed2:
        current_pos += 1
        if current_pos - num_removed2 >= size:
            end_not_reached = False
        continue
    w = copy.deepcopy(best_weights2)
    w[0][:,current_pos] = 0
    w[1][current_pos] = 0
    w[2][current_pos,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    if 0.1*(na - oa) + 0.9*(ol - nl) > best_change:
        best_change = 0.1*(na - oa) + 0.9*(ol - nl)
        print(best_change)
        best_pos = current_pos
        improved = True
        best_acc = na
        best_loss = nl
        print("Found something better")
    current_pos += 1
    if current_pos - num_removed2 >= size:
        end_not_reached = False
    if current_pos%20 == 0:
        print("Did 20 iterations")

tester_model.set_weights(best_weights2)
loss2, acc2 = tester_model.evaluate(x_test, y_test, verbose=2)


# %%
for i in nodes_removed2:
    best_weights2[0][:,i] = np.random.normal(0, 2/np.sqrt(28*28 + 128), 784)
    best_weights2[1][i] = 0
    best_weights2[2][i,:] = np.random.normal(0, 2/np.sqrt(138), 10)


# %%
new_model = tf.keras.models.Sequential()
new_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
new_model.add(tf.keras.layers.Dense(128, activation='relu'))
new_model.add(tf.keras.layers.Dense(10, activation='softmax'))

new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

new_model.set_weights(best_weights2)

new_model.evaluate(x_test, y_test, verbose=2)
new_model.fit(x_train, y_train, epochs=5)
new_model.evaluate(x_test, y_test, verbose=2)


# %%
loss, acc = model.evaluate(x_test, y_test, verbose=2)
end_not_reached = True
improved = False
size = 128
tol = -1e-30
current_pos = 0
best_pos = -1
best_change = tol
original2 = model.get_weights()
bas2 = [acc]
bls2 = [loss]
best_weights2 = model.get_weights()
nodes_removed2 = []
best_acc = 0
best_loss = 1e20
l = loss
a = acc
num_removed2 = 0

for i in range(128):
    w = copy.deepcopy(original2)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    change = 0.*(na - a) + 1.0*(l - nl)
    print(f"Node {i}:", change)
    if change > tol:
        nodes_removed2 += [i]
        num_removed2 += 1
        
for i in nodes_removed2:
    best_weights2[0][:,i] = 0
    best_weights2[1][i] = 0
    best_weights2[2][i,:] = 0

tester_model.set_weights(best_weights2)
loss2, acc2 = tester_model.evaluate(x_test, y_test, verbose=2)


# %%
l, a = model.evaluate(x_test, y_test, verbose=2)
or_weights = model.get_weights()
size = 128
worst_remove = -1
wc = 0
w2 = model.get_weights()
for i in range(128):
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.*(na - a) + 1.0*(l - nl))
    if 0.*(na - a) + 1.0*(l - nl) < wc:
        worst_remove = i
        wc = (l - nl)
w2[0][:,worst_remove] = 0
w2[1][worst_remove] = 0
w2[2][worst_remove,:] = 0
tester_model.set_weights(w2)
loss2, acc2 = tester_model.evaluate(x_test, y_test, verbose=2)
l = loss2
a = acc2
for i in range(128):
    w = copy.deepcopy(w2)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model.set_weights(w)
    nl, na = tester_model.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.*(na - a) + 1.0*(l - nl))

# %% [markdown]
# # Testing on higher node counts

# %%
size = 1024


# %%
model2 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(size, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=10, batch_size=None)


# %%
model2.fit(x_train, y_train, epochs=5, batch_size=1024)


# %%
model2.fit(x_train, y_train, epochs=5, batch_size=256)


# %%
model2.fit(x_train, y_train, epochs=5, batch_size=32)


# %%
tester_model2 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(size, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
tester_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# %%
model2.evaluate(x_test, y_test, verbose=2, batch_size=256)


# %%
l, a = model2.evaluate(x_test, y_test, verbose=2, batch_size=256)
or_weights = model2.get_weights()
tol_low = -1e-5
tol_high = 1e-5
num_zeros, num_worse, num_important = (0, 0, 0)
z = []
wr = []
imp = []
for i in range(size):
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model2.set_weights(w)
    nl, na = tester_model2.evaluate(x_test, y_test, verbose=0, batch_size=256)
    print(f"Node {i}:", 1.*(na - a) + 0*(l - nl))
    change = l - nl
    if change <= tol_high and change >= tol_low:
        num_zeros += 1
        z += [i]
    elif change > 0:
        num_worse += 1
        wr += [i]
    else:
        num_important += 1
        imp += [i]


# %%
print("Zero Nodes:", num_zeros)
print("Worse Nodes:", num_worse)
print("Important Nodes:", num_important)


# %%
print("######## IMPORTANT NODES ########")
for i in imp:
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model2.set_weights(w)
    nl, na = tester_model2.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.*(na - a) + 1.0*(l - nl))


# %%
print("######## WORSE NODES ########")
tot = 0
for i in wr:
    w = copy.deepcopy(or_weights)
    w[0][:,i] = 0
    w[1][i] = 0
    w[2][i,:] = 0
    tester_model2.set_weights(w)
    nl, na = tester_model2.evaluate(x_test, y_test, verbose=0)
    print(f"Node {i}:", 0.*(na - a) + 1.0*(l - nl))
    tot += (l - nl)
print(tot)
print(tot / num_worse)


# %%
loss, acc = model2.evaluate(x_test, y_test, verbose=2, batch_size=512)
end_not_reached = True
improved = False
tol = -1e-5
current_pos = 0
best_pos = -1
best_change = tol
original2 = model2.get_weights()
bas2 = [acc]
bls2 = [loss]
best_weights2 = model2.get_weights()
nodes_removed2 = []
best_acc = 0
best_loss = 1e20
ol = loss
oa = acc
num_removed2 = 0
while end_not_reached or improved:
    if not(end_not_reached):
        end_not_reached = True
        improved = False
        current_pos = 0
        size -= 1
        nodes_removed2 += [best_pos]
        best_weights2[0][:,best_pos] = 0
        best_weights2[1][best_pos] = 0
        best_weights2[2][best_pos,:] = 0
        best_pos = -1
        #tol -= best_change
        ol = best_loss
        oa = best_acc
        bas2 += [best_acc]
        bls2 += [best_loss]
        print("Improvement has occured!! Accuracy:", best_acc, "--- Loss:", best_loss, '--- Change:', best_change, '--- New tol:', tol)
        best_change = tol
        num_removed2 += 1
    if current_pos in nodes_removed2:
        current_pos += 1
        if current_pos - num_removed2 >= size:
            end_not_reached = False
        continue
    w = copy.deepcopy(best_weights2)
    w[0][:,current_pos] = 0
    w[1][current_pos] = 0
    w[2][current_pos,:] = 0
    tester_model2.set_weights(w)
    nl, na = tester_model2.evaluate(x_test, y_test, verbose=0, batch_size=512)
    if 0.3*(na - oa) + 0.7*(ol - nl) > best_change:
        best_change = 0.3*(na - oa) + 0.7*(ol - nl)
        print(best_change)
        best_pos = current_pos
        improved = True
        best_acc = na
        best_loss = nl
        print("Found something better")
    current_pos += 1
    if current_pos - num_removed2 >= size:
        end_not_reached = False
    if current_pos%200 == 0:
        print("Did 200 iterations")

tester_model2.set_weights(best_weights2)
loss2, acc2 = tester_model2.evaluate(x_test, y_test, verbose=2)


# %%
num_removed2

# %% [markdown]
# ## Junk + Testing

# %%
type(model.get_weights()[0][:,0])


# %%
old = model.get_weights()
old[0][:,0] = 0


# %%
old[1][0] = 0


# %%
old[2][0,:] = 0


# %%
np.shape(old[2])


# %%
model.set_weights(old)


# %%
model.evaluate(x_test, y_test, verbose=2)


# %%
model.fit(x_train, y_train, epochs=5)


# %%
model.evaluate(x_test, y_test, verbose=2)


# %%
new = model.get_weights()


# %%


