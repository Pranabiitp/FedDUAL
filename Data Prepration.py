


import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset
cifar10_data = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10_data.load_data()

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(train_images))
train_images_shuffled = train_images[shuffle_indices]
train_labels_shuffled = train_labels[shuffle_indices]

# Number of clients
num_clients = 100

# Number of classes in CIFAR-10
num_classes = 10
alpha=0.01
# Simulate heterogeneous partition using Dirichlet distribution
# Here, we'll assume equal proportions for simplicity
proportions = np.random.dirichlet(np.ones(num_clients) * alpha, size=num_classes)

# Allocate data to clients
client_data_indices = [[] for _ in range(num_clients)]
for class_label in range(num_classes):
    num_samples_per_class = np.sum(train_labels_shuffled == class_label)
    for client_idx in range(num_clients):
        num_samples_for_client = int(proportions[class_label, client_idx] * num_samples_per_class)
        # Randomly select data indices for the client, ensuring at least one sample per client
        if num_samples_for_client == 0:
            num_samples_for_client = 1
        selected_indices = np.random.choice(
            np.where(train_labels_shuffled[:, 0] == class_label)[0],
            size=num_samples_for_client,
            replace=False
        )
        client_data_indices[client_idx].extend(selected_indices)

# Initialize lists to store data and labels for each client
client_train_data = []
client_train_labels = []

# Extract data and corresponding one-hot labels for each client
for client_indices in client_data_indices:
    client_data_samples = train_images_shuffled[client_indices]
    client_data_labels = train_labels_shuffled[client_indices]

    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(client_data_labels, num_classes)

    # Append data and labels for this client to the list
    client_train_data.append(client_data_samples)
    client_train_labels.append(one_hot_labels)

# Naming the variables as train1, label1, train2, label2, and so on for all clients
for i in range(len(client_train_data)):
    globals()[f"train{i+1}"] = client_train_data[i]
    globals()[f"label{i+1}"] = client_train_labels[i]

# Print out the number of samples allocated to each client for debugging
for i in range(num_clients):
    print(f"Client {i+1}: {len(client_data_indices[i])} samples")





import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-100 dataset
cifar100_data = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100_data.load_data()

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(train_images))
train_images_shuffled = train_images[shuffle_indices]
train_labels_shuffled = train_labels[shuffle_indices]

# Number of clients
num_clients = 100

# Number of classes in CIFAR-100
num_classes = 100
alpha=0.01
# Simulate heterogeneous partition using Dirichlet distribution
# Here, we'll assume equal proportions for simplicity
proportions = np.random.dirichlet(np.ones(num_clients) * alpha, size=num_classes)

# Allocate data to clients
client_data_indices = [[] for _ in range(num_clients)]
for class_label in range(num_classes):
    num_samples_per_class = np.sum(train_labels_shuffled == class_label)
    for client_idx in range(num_clients):
        num_samples_for_client = int(proportions[class_label, client_idx] * num_samples_per_class)
        if num_samples_for_client == 0:
            num_samples_for_client = 1
        # Randomly select data indices for the client
        selected_indices = np.random.choice(
            np.where(train_labels_shuffled[:,0] == class_label)[0],
            size=num_samples_for_client,
            replace=False
        )
        client_data_indices[client_idx].extend(selected_indices)

# Now client_data_indices contains the indices of data samples allocated to each client
# Initialize lists to store data and labels for each client
client_train_data = []
client_train_labels = []

# Extract data and corresponding one-hot labels for each client
for client_indices in client_data_indices:
    client_data_samples = train_images_shuffled[client_indices]
    client_data_labels = train_labels_shuffled[client_indices]
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(client_data_labels, num_classes)
    
    # Append data and labels for this client to the list
    client_train_data.append(client_data_samples)
    client_train_labels.append(one_hot_labels)

# Naming the variables as train1, label1, train2, label2, and so on for all 10 clients
for i in range(len(client_train_data)):
    globals()[f"train{i+1}"] = client_train_data[i]
    globals()[f"label{i+1}"] = client_train_labels[i]





import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Fashion MNIST dataset
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(train_images))
train_images_shuffled = train_images[shuffle_indices]
train_labels_shuffled = train_labels[shuffle_indices]

# Number of clients
num_clients = 100

# Number of classes in Fashion MNIST
num_classes = 10
alpha=0.01
# Simulate heterogeneous partition using Dirichlet distribution
# Here, we'll assume equal proportions for simplicity
proportions = np.random.dirichlet(np.ones(num_clients) * alpha, size=num_classes)

# Allocate data to clients
client_data_indices = [[] for _ in range(num_clients)]
for class_label in range(num_classes):
    num_samples_per_class = np.sum(train_labels_shuffled == class_label)
    for client_idx in range(num_clients):
        num_samples_for_client = int(proportions[class_label, client_idx] * num_samples_per_class)
        # Randomly select data indices for the client
        if num_samples_for_client == 0:
            num_samples_for_client = 1
        selected_indices = np.random.choice(
            np.where(train_labels_shuffled == class_label)[0],
            size=num_samples_for_client,
            replace=False
        )
        client_data_indices[client_idx].extend(selected_indices)

# Now client_data_indices contains the indices of data samples allocated to each client
# Initialize lists to store data and labels for each client
client_train_data = []
client_train_labels = []

# Extract data and corresponding one-hot labels for each client
for client_indices in client_data_indices:
    client_data_samples = train_images_shuffled[client_indices]
    client_data_labels = train_labels_shuffled[client_indices]
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(client_data_labels, num_classes)
    
    # Append data and labels for this client to the list
    client_train_data.append(client_data_samples)
    client_train_labels.append(one_hot_labels)

# Naming the variables as train1, label1, train2, label2, and so on for all 10 clients
for i in range(len(client_train_data)):
    globals()[f"train{i+1}"] = client_train_data[i]
    globals()[f"label{i+1}"] = client_train_labels[i]

# Now, client_train_data[i] and client_train_labels[i] contain the data and corresponding one-hot labels for the i-th client

