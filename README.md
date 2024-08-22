## FedDUAL: A Dual-Strategy with Adaptive Loss and Dynamic Aggregation for Mitigating Data Heterogeneity in Federated Learning
This repository contains the code for the paper "FedDUAL: A Dual-Strategy with Adaptive Loss and Dynamic Aggregation for Mitigating Data Heterogeneity in Federated Learning".

# Dependencies
- Tensorflow = 2.10.0
- scikit-learn = 1.3.2

# Data Preparing
To divide the dataset into the required no. of clients, run Data Prepration.py and choose the required dataset (CIFAR10, CIFAR100 or FMNIST) and then change the degree of heterogenity (alpha) as required. you will get the desired distribution for each client.

# Model Structure
To choose the appropriate  model, run Models.py and choose the required model for each of the dataset.

# Run FedDUAL
After done with above process, you can run the FedDUAL, our proposed method by running FedDUAL.py.

# Evaluation
After federated training, run Evaluation.py to acess the evaluation metrics such as accuracy, precision, recall etc.

