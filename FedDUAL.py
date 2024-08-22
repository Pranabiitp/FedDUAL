


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)






def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict






def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss







import tensorflow as tf

def avg_weights(scaled_weight_list):
    '''Return the average of the listed scaled weights.'''
    num_clients = len(scaled_weight_list)
    
    if num_clients == 0:
        return None  # Handle the case where the list is empty
        
    avg_grad = list()
    
    # Get the sum of gradients across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / num_clients
        avg_grad.append(layer_mean)
        
    return avg_grad





import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
from scipy.stats import wasserstein_distance

def kl_divergence(p, q):
    # Ensure probabilities are non-zero to avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

def weights_to_probs(weights):
    # Convert model weights to probability distributions using softmax
    flat_weights = np.concatenate([w.flatten() for w in weights])
    probs = np.exp(flat_weights) / np.sum(np.exp(flat_weights))
    return probs

def adaptive_loss_function(lambda_, local_loss, global_probs, local_probs):
    # Compute KL divergence
    divergence = kl_divergence(local_probs, global_probs)
    
    # Compute adaptive loss
    return (1 - lambda_) * local_loss + lambda_ * divergence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

comms_round = 500  # Number of global epochs
acc3 = []
loss3 = []
train_acc_clients = [[], [], [], []]  # List of lists for training accuracy for each client
val_acc_clients = [[], [], [], []]    # List of lists for validation accuracy for each client
best_acc = 0
best_weights = None

# Function to compute the gradient norms for each layer
def compute_gradient_norms(model, data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_norms = [tf.norm(grad).numpy() for grad in gradients]
    return gradients, gradient_norms

# Function to compute Wasserstein Barycenter
def wasserstein_barycenter(distributions, weights, num_iter=1000, epsilon=0.00001):
    # Initial guess for the barycenter
    barycenter = np.mean(distributions, axis=0)

    for _ in range(num_iter):
        new_barycenter = np.zeros_like(barycenter)
        total_weight = np.zeros_like(barycenter)

        for dist, weight in zip(distributions, weights):
            gamma = np.exp(-wasserstein_distance(barycenter, dist) / epsilon)
            new_barycenter += weight * gamma * dist
            total_weight += weight * gamma

        barycenter = new_barycenter / (total_weight + 1e-10)  # Adding a small epsilon to avoid division by zero

    return barycenter

for comm_round in range(comms_round):
    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    global_probs = weights_to_probs(global_weights)

    # Initial lists to collect local model weights and gradient norms
    local_weight_list = []
    local_gradients_list = []
    gradient_norm_list = []

    # Randomize and select 2 client names for this round
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)
    client_names = client_names[:10]  # Selecting 2 random clients

    for i, client in enumerate(client_names):
        local_model = build_lenet()

        # Set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        print(client)
        
        # Fit local model with client's data
        data, labels = np.array(clients_batched[client][0]), np.array(clients_batched[client][1])
        val_data = np.array(test_batched[client][0])
        val_labels = np.array(test_batched[client][1])
        # Calculate the local accuracy
        local_acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(local_model.predict(data), axis=1))

        # Calculate the global accuracy on the local client's data
        global_acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(global_model.predict(data), axis=1))
        

        # Convert local model weights to probabilities
        local_probs = weights_to_probs(local_model.get_weights())

#         # Compile the local model with the custom adaptive loss function
        local_model.compile(optimizer=tf.keras.optimizers.Adam(), 
                            loss=lambda y_true, y_pred: adaptive_loss_function(
                                sigmoid(local_acc - global_acc),
                                tf.keras.losses.categorical_crossentropy(y_true, y_pred),
                                global_probs,
                                local_probs
                            ),
                            metrics=['accuracy'])
        
        # Train the local model for one epoch
        history = local_model.fit(
            data,
            labels,
            validation_data=(val_data, val_labels),
            epochs=3,
            batch_size=10,  # Ensure batch size matches data size
            verbose=2
        )

        # Get the local model weights and compute gradient norms
        gradients, gradient_norms = compute_gradient_norms(local_model, data, labels)
        
        # Calculate the local accuracy
        local_acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(local_model.predict(data), axis=1))

        # Calculate the global accuracy on the local client's data
        global_acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(global_model.predict(data), axis=1))
        
        # Add to the lists
        local_weight_list.append(local_model.get_weights())
        local_gradients_list.append(gradients)
        gradient_norm_list.append(gradient_norms)

    # Calculate the Wasserstein Barycenter for the last layer gradients
    last_layer_gradients = [grads[-1].numpy().flatten() for grads in local_gradients_list]
    second_last_layer_gradients = [grads[-2].numpy().flatten() for grads in local_gradients_list]
    weights = np.ones(len(client_names)) / len(client_names)  # Equal weights for simplicity
    barycenter_gradients = wasserstein_barycenter(last_layer_gradients, weights)
    barycenter_second_last_layer = wasserstein_barycenter(second_last_layer_gradients, weights)
    
    # Update the last layer's weights
    average_weights = []
    for i in range(len(global_weights)):
        if i == len(global_weights) - 1:  # Last layer
            average_weights.append(global_weights[i] - barycenter_gradients.reshape(global_weights[i].shape))
        elif i == len(global_weights) - 2:  # Second-to-last layer
            average_weights.append(global_weights[i] - barycenter_second_last_layer.reshape(global_weights[i].shape))   
                
        else:
            # Standard federated averaging for other layers
            avg_weight = np.mean([local_weights[i] for local_weights in local_weight_list], axis=0)
            average_weights.append(avg_weight)

    # Update the global model with the average weights
    global_model.set_weights(average_weights)

    # Test the global model and print out metrics after each communications round
    global_acc, global_loss = test_model(test, label, global_model, comm_round)
    acc3.append(global_acc)
    loss3.append(global_loss)
    
    # Plotting the results (optional)
    import matplotlib.pyplot as plt
    plt.plot(acc3)
    plt.show()

    


