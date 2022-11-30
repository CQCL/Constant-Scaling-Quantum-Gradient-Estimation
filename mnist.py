import torch
from torch import nn
import os
import sys
import pennylane as qml
from pennylane import numpy as np

from spsb_diff import spsb_diff
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torchmetrics import Accuracy
from tqdm import trange

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))

# np.random.seed(42)
# torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_DEVICE = torch.device('cuda:0')

N_SAMPLES = 1000
N_QUBITS = 4
N_LAYERS = 3
epochs = 8
N_ITERATIONS = 5

differentiator = sys.argv[1]

if differentiator == "backprop":
    diff_method = "backprop"
elif differentiator == "parameter-shift":
    N_EPOCHS = epochs
    diff_method = "parameter-shift"
elif differentiator == "spsb":
    diff_method = spsb_diff
    N_EPOCHS = epochs * (2*N_QUBITS*N_LAYERS + 1)//(3)
else:
    raise ValueError("Differentiator not recognised.")


BATCH_SIZE = 50
DEVICE = torch.device('cpu')

# Create transform to downside MNIST images to 4x4
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((4, 4)), transforms.Normalize((0.1307,), (0.3081,))])
mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)


# Get only 6s and 3s
idx = np.where((mnist.targets == 6) | (mnist.targets == 3))[0]
mnist.data = mnist.data[idx]
mnist.targets = mnist.targets[idx]

# Limit to N_SAMPLES
mnist.data = mnist.data[:N_SAMPLES]
mnist.targets = mnist.targets[:N_SAMPLES]

#Change labels to 
mnist.targets[mnist.targets == 6] = 0
mnist.targets[mnist.targets == 3] = 1



dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, diff_method=diff_method, interface="torch")
def circuit(inputs, weights):

    # Encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Classifier
    for l in range(N_LAYERS):
        layer_params = weights[N_QUBITS*l:N_QUBITS*(l+1)]
        for i in range(N_QUBITS):
            qml.CRZ(layer_params[i], wires=[i, (i+1)%N_QUBITS])
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class Model(nn.Module):
    def __init__(self, n_wires, n_layers) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        weight_shapes = {"weights": (self.n_layers * self.n_wires,)}
        self.quanv = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.lin = nn.Linear(4*2*2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        bsz = x.shape[0]
        size = 4
        x = x.view(bsz, size, size)

        data_list = []

        for j in range(0, size, 2):
            for k in range(0, size, 2):
                data = torch.transpose(torch.cat((x[:, j, k], x[:, j, k+1], x[:, j+1, k], x[:, j+1, k+1])).view(4, bsz), 0, 1)
                q_results = self.quanv(data)

                data_list.append(q_results.view(bsz, 4))

        x = torch.cat(data_list, dim=1).float()

        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        x = self.softmax(x)
        return x



def get_losses(LEARNING_RATE):

    losses_all = []
    accs_all = []
    evals_all = []

    for _ in trange(N_ITERATIONS, desc='Iterations'):

        trainloader = torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

        model = Model(N_QUBITS, N_LAYERS)
        model.to(DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss = nn.CrossEntropyLoss()
        acc = Accuracy().to(DEVICE)

        initial_loss = 0.
        initial_acc = 0.

        # Get initial loss on whole dataset
        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.squeeze().to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            l = loss(output, target)
            initial_loss += l.item()
            initial_acc += output.argmax(dim=1).eq(target).sum().item()
        initial_loss /= len(trainloader)
        initial_acc /= len(trainloader)

        losses = [initial_loss]
        avg_accs = [initial_acc]

        # Circuit evals
        init_circuit_evals = dev.num_executions
        circuit_evals = [0]

        for epoch in trange(N_EPOCHS, desc="Epochs"):
            # losses = []
            running_acc = 0
            for batch_idx, (data, target) in enumerate(trainloader):
                data = data.squeeze().to(DEVICE)
                target = target.to(DEVICE)
                out = model(data)

                l = loss(out, target)
                losses.append(l.item())

                opt.zero_grad()
                l.backward()
                opt.step()

                accuracy = acc(out, target)
                running_acc += len(target) * accuracy.item()
                circuit_evals.append(dev.num_executions - init_circuit_evals)

            avg_accs.append(running_acc / len(trainloader.dataset))

            print(f"Epoch {epoch+1}/{N_EPOCHS}")

        losses_all.append(losses)
        accs_all.append(avg_accs)
        evals_all.append(circuit_evals)

    return losses_all, accs_all, evals_all



def calculate(LEARNING_RATE):
    # LEARNING_RATE = hps['LEARNING_RATE']
    losses_all, accs_all, circuit_evals = get_losses(LEARNING_RATE)
    losses_all = np.array(losses_all)
    accs_all = np.array(accs_all)
    np.save(f"data_quanv/new_fixed_diffseed_losses_all_4_{epochs}_{LEARNING_RATE}_{differentiator}_{BATCH_SIZE}", losses_all)
    np.save(f"data_quanv/new_fixed_diffseed_accs_all_4_{epochs}_{LEARNING_RATE}_{differentiator}_{BATCH_SIZE}", accs_all)
    np.save(f"data_quanv/new_fixed_diffseed_evals_4_{epochs}_{LEARNING_RATE}_{differentiator}_{BATCH_SIZE}", circuit_evals)



lr = float(sys.argv[2])

print(differentiator, lr)

calculate(lr)
