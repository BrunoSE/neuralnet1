# Import the needed libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt


# Load dataset
names = ['es_feriado', 'es_dia_pago', 'es_dia_juega_chile', 'es_segunda_quincena', 'es_falla']

df = pd.read_csv('fallas_training.csv', names=names)
df_test = pd.read_csv('fallas_test.csv', names=names)


# ## Separate the majority and minority classes
df_minority = df[df['es_falla'] == 1]
df_majority = df[df['es_falla'] == 0]

# ## Now, downsamples majority labels equal to the number of samples in the minority class

df_majority = df_majority.sample(len(df_minority), random_state=0)

# ## concat the majority and minority dataframes
df = pd.concat([df_majority, df_minority])

# # Shuffle the dataset to prevent the model from getting biased by similar samples
df = df.sample(frac=1, random_state=0)
Xtotal = df.drop("es_falla", axis=1)
ytotal = df.es_falla
Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtotal, ytotal, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Encode target values into binary ('one-hot' style) representation
ytrain = pd.get_dummies(ytrain)
yvalid = pd.get_dummies(yvalid)

print("Total filas entrenamiento y testeo:", len(Xtrain.index), len(Xvalid.index))
# Create and train a tensorflow model of a neural network


def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    # tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(len(Xtrain.index), 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(len(ytrain.index), 2), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 2), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain.values, y: ytrain.values}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2


# Run the training for 3 different network architectures: (4-5-3) (4-10-3) (4-20-3)

# Plot the loss function over iterations
comparar_numero_nodos_ocultos = True
num_iters = 250

if comparar_numero_nodos_ocultos:
    num_hidden_nodes = [2, 4, 6, 8]  # nodos de la capa intermedia de red neuronal
    loss_plot = {}
    weights1 = {}
    weights2 = {}
    for nhn in num_hidden_nodes:
        loss_plot[nhn] = []
        weights1[nhn] = None
        weights2[nhn] = None

    plt.figure(figsize=(12, 8))
    for hidden_nodes in num_hidden_nodes:
        weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
        plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: NX-%d-NY" % hidden_nodes)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

else:
    num_hidden_nodes = [5]
    loss_plot = {5: []}
    weights1 = {5: None}
    weights2 = {5: None}
    for hidden_nodes in num_hidden_nodes:
        weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)


# Evaluate models on the test set
Xtest = df_test.drop("es_falla", axis=1)
ytest = df_test.es_falla
ytest = pd.get_dummies(ytest)

X = tf.placeholder(shape=(len(Xtest.index), 4), dtype=tf.float64, name='X')
y = tf.placeholder(shape=(len(ytest.index), 2), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

    # Calculate the prediction accuracy

    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) for estimate, target in zip(y_est_np, ytest.values)]
    accuracy = 100 * sum(correct) / len(correct)
    print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))
