import torch
import pennylane as qml
from pennylane import numpy as np

from spsb_diff import spsb_diff
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
GPU_DEVICE = torch.device('cuda:0')

N_ITERATIONS = 10
N_SAMPLES = 100
N_LAYERS = 3
BATCH_SIZE = 25
LEARNING_RATE = 0.01

class Rescale(torch.nn.Module):
    def forward(self, x):
        return (x+1)/2

def get_losses(differentiator, n_features, epochs):

    def circuit_spsb(inputs, weights):
        for i in range(len(inputs)):
            qml.RX(inputs[i], wires=i)
            qml.Hadamard(wires=i)

        for l in range(N_LAYERS):
            layer_params = weights[n_features*l: n_features*(l+1)]
            for i in range(n_features):
                qml.CRZ(layer_params[i], wires=[i, (i+1)%n_features])
            for i in range(n_features):
                qml.Hadamard(wires=i)

        return qml.expval(qml.PauliZ(n_features-1))

    if differentiator == "spsb":
        diff_method = spsb_diff
    elif differentiator == "ps":
        diff_method = "parameter-shift"

    losses_all = []
    accs_all = []

    for _ in trange(N_ITERATIONS, desc="Iterations", leave=False, position=1):

        # Init model
        dev = qml.device('lightning.gpu', wires=n_features)
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit_spsb)
        weight_shapes = {"weights": (n_features*N_LAYERS,)}
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        model = torch.nn.Sequential(qlayer, Rescale()).to(GPU_DEVICE)

        # Init dataset
        X, Y = make_classification(N_SAMPLES, n_features=n_features)
        # scale normal X values to interval [0,2pi]
        X = MinMaxScaler(feature_range=(0, 2*np.pi)).fit_transform(X)
        X = torch.tensor(X, dtype=torch.float).to(GPU_DEVICE)
        Y = torch.tensor(Y, dtype=torch.float).to(GPU_DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_f = torch.nn.BCELoss()
        acc = Accuracy().to(GPU_DEVICE)

        # Train
        data_set = torch.utils.data.TensorDataset(X, Y)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  drop_last=False)

        pred_init = model(X).squeeze(-1)
        losses = [loss_f(pred_init, Y).item()]
        avg_accs = [acc(pred_init, Y.int()).item()]

        init_circuit_evals = dev.num_executions
        circuit_evals =[0]
        for _ in trange(epochs, desc="Epochs", leave=False, position=2):
            running_acc = 0
            for x, y in data_loader:
                pred = model(x).squeeze(-1)
                loss = loss_f(pred, y)
                accuracy = acc(pred, y.int())
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_acc += len(y) * accuracy.item()
                losses.append(loss.item())
                circuit_evals.append(dev.num_executions - init_circuit_evals)
            avg_acc = running_acc / len(data_set)
            avg_accs.append(avg_acc)

        losses_all.append(losses)
        accs_all.append(avg_accs)

    return losses_all, accs_all, circuit_evals

N_FEATURES = [5, 10, 15]
PS_EPOCHS = [300, 200, 100]

for ps_eps, n_feat in tqdm(zip(PS_EPOCHS, N_FEATURES), desc="Features", leave=False, position=0):
    spsb_eps = (ps_eps * ((1+n_feat*N_LAYERS*4)*N_SAMPLES))//(3*N_SAMPLES)
    file_template = "data_initital_exp_wO_fc/wofc_{}_{}_niter{}_nfeatures{}_nlayers{}_batchsize{}_nepochs{}_lr{}.npy"
    spsb_losses_name = file_template.format("spsb", "loss", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, spsb_eps, LEARNING_RATE)
    spsb_accs_name = file_template.format("spsb", "acc", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, spsb_eps, LEARNING_RATE)
    spsb_evals_name = file_template.format("spsb", "evals", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, spsb_eps, LEARNING_RATE)
    ps_losses_name = file_template.format("ps", "loss", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, ps_eps, LEARNING_RATE)
    ps_accs_name = file_template.format("ps", "acc", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, ps_eps, LEARNING_RATE)
    ps_evals_name = file_template.format("ps", "evals", N_ITERATIONS, n_feat, N_LAYERS, BATCH_SIZE, ps_eps, LEARNING_RATE)

    losses_spsb, accs_spsb, evals_spsb = get_losses("spsb", n_feat, spsb_eps)
    np.save(spsb_losses_name, losses_spsb)
    np.save(spsb_accs_name, accs_spsb)
    np.save(spsb_evals_name, evals_spsb)

    losses_ps, accs_ps, evals_ps= get_losses("ps", n_feat, ps_eps)
    np.save(ps_losses_name, losses_ps)
    np.save(ps_accs_name, accs_ps)
    np.save(ps_evals_name, evals_ps)
