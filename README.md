# Constant-Scaling Quantum Gradient Estimation

This repository contains the code to reproduce the results of the paper "Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning" by Thomas Hoffmann and Douglas Brown.

# Structure

- `random_dps_test.py`: contains the experiment for random datapoints with a fully connected layer with sigmoid activation as the output layer

- `random_dps_test_wo_fc`: contains the experiment for random datapoints with the classification coming from the rescaled Z-expectation of a single qubit

- `mnist.py`: contains the experiment for the `quanvolutional` neural network based on [Henderson et al. 2019]

- `random_dps_plots`: contains the code to make Fig. 2 (results of the random datapoints experiment)

- `lr_plots.py`: contains the code to make Fig. 3 (results of investigating the effect of learning rate on the convergence of each task)

- `/data_...`: contains data for the experiment in question 

- `/figs` contains figures used in the report

- `spsb_diff` and `spsb_diff_smoothing`: contains the `spsb` differentiator (with optional Jacobian smoothing) which can be imported to use in the `diff_method` argument in a PennyLane `qnode` as below:
    ```
    import pennylane as qml
    from spsb_diff import spsb_diff
    dev = qml.device('default.qubit', wires=1)

    @qml.qnode(dev, diff_method=spsb_diff)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    ```

