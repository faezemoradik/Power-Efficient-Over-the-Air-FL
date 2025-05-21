# Power-Efficient Over-the-Air Aggregation with Receive Beamforming for Federated Learning

This repository contains the implementation for **Power-Efficient Over-the-Air Aggregation with Receive Beamforming for Federated Learning**.


In order to use the code, please follow these steps:

## 1- Install requirements:

    pip install -r requirements.txt

## 2- Set the arguments in the myrun.py file:

    1- learning_rate: The learning rate, γ

    2- batch_size: The size of the batch for each device

    3- num_epoch: Maximum number of training epochs

    4- dataset: Dataset name; can be either "MNIST" or "CIFAR-10"

    5- myseed: Seed for generating the wireless channel, receiver noise, and batch sampling in each communication round

    6- method: Name of the method for beamforming design; choose one of the following:
       "GSDS", "MMSE", "BoundedMSE", "PoMFL", or "PoMFL-ImCSI"

    7- P_zero: Maximum power limit for devices, P₀

    8- alpha: A constant in the PoMFL (or PoMFL-ImCSI) method, α

    9- delta: A constant in the PoMFL (or PoMFL-ImCSI) method, δ

    10- beta: A constant in the PoMFL (or PoMFL-ImCSI) method, β

    11- MSE_bound: A constant in the Bounded MSE method, η

    12- epsilon: The normalized error for channel estimation, 0 ≤ ε ≤ 1.
    
        - If set to 0, no error exists and the true and estimated channels are equal.
        - If greater than 0, the estimated channel differs from the true one.

## 3- Run the myrun.py file:

    python myrun.py


Please note that when running a specific method, the parameters related to other methods can be set to any value — they will not affect the execution of the selected method.

Thank you for your attention!
