# ==============================================================================
# IMPORTS
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.models import resnet18
from torchvision import transforms
import flwr as fl
from flwr.common import ndarrays_to_parameters
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for a cleaner output.
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. DEVICE CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# ==============================================================================
# 2. MODEL DEFINITION (USING RESNET-18)
# ==============================================================================
def load_model():
    """Load a ResNet18 model and adapt it for our 10-class problem."""
    # We load a ResNet18 model without pre-trained weights.
    # Training from scratch is common in FL to avoid bias from the pre-training dataset.
    model = resnet18(weights=None)

    # The final layer of ResNet18 is a fully connected layer ('fc').
    # We get the number of input features for this layer.
    num_ftrs = model.fc.in_features

    # We replace the final layer with a new one that has 10 output units (for our 10 classes).
    model.fc = nn.Linear(num_ftrs, 10)

    # Move the model to the specified device (CPU or GPU).
    return model.to(DEVICE)

# ==============================================================================
# 3. DATA LOADING & TRANSFORMATION
# ==============================================================================
def load_data(cid: int):
    """Load a unique dataset for each client."""
    # We use a standard transform that resizes images and converts grayscale to 3 channels.
    # This is necessary because ResNet expects 3-channel input images.
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if cid == 0:
        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_size = len(dataset) // 2
        trainset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    elif cid == 1:
        trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    elif cid == 2:
        trainset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    else:
        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_size = len(dataset) // 2
        _, second_half = random_split(dataset, [train_size, len(dataset) - train_size])
        client_3_size = len(second_half) // 2
        if cid == 3:
            trainset, _ = random_split(second_half, [client_3_size, len(second_half) - client_3_size])
        else:
            _, trainset = random_split(second_half, [client_3_size, len(second_half) - client_3_size])

    # The test set is always CIFAR10 for a consistent evaluation benchmark.
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader

# ==============================================================================
# 4. TRAINING & TESTING UTILITIES
# ==============================================================================
def train(net, trainloader, epochs):
    """Train the model on the client's local data."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Evaluate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

# ==============================================================================
# 5. FLOWER CLIENT IMPLEMENTATION
# ==============================================================================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=7) # 7 local epochs
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

# ==============================================================================
# 6. CLIENT FACTORY FUNCTION
# ==============================================================================
def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client instance for a given client ID."""
    net = load_model()
    trainloader, testloader = load_data(int(cid))
    return FlowerClient(net, trainloader, testloader).to_client()

# ==============================================================================
# 7. METRICS AGGREGATION FUNCTION
# ==============================================================================
def weighted_average(metrics):
    """Aggregate evaluation results from multiple clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# ==============================================================================
# 8. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Create an instance of the ResNet model to extract initial parameters
    net = load_model()
    initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])

    # Define the federated learning strategy (FedAdam)
    strategy = fl.server.strategy.FedAdam(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=4,
        min_evaluate_clients=2,
        min_available_clients=4,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start the Flower simulation for 50 rounds
    print("Starting Federated Learning Simulation with ResNet18 for 50 rounds...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=50), # 50 rounds
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25 if DEVICE.type == "cuda" else 0},
    )
    print("Federated Learning Simulation finished.")

    # ==========================================================================
    # 9. VISUALIZATION
    # ==========================================================================
    if history.metrics_distributed:
        print("Plotting results...")
        
        loss_distributed = history.losses_distributed
        accuracy_distributed = history.metrics_distributed["accuracy"]

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        rounds = [x[0] for x in loss_distributed]
        loss = [x[1] for x in loss_distributed]
        color = 'tab:red'
        ax1.set_xlabel('Federated Round', fontsize=14)
        ax1.set_ylabel('Centralized Loss', color=color, fontsize=14)
        ax1.plot(rounds, loss, color=color, marker='o', linestyle='dashed', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2 = ax1.twinx()
        
        acc_rounds = [x[0] for x in accuracy_distributed]
        accuracy = [x[1] * 100 for x in accuracy_distributed]
        color = 'tab:blue'
        ax2.set_ylabel('Centralized Accuracy (%)', color=color, fontsize=14)
        ax2.plot(acc_rounds, accuracy, color=color, marker='x', linestyle='solid', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title("Federated Learning Performance (ResNet18, 50 Rounds)", fontsize=16)
        plt.xticks(range(0, 51, 5))
        plt.legend(loc='best')
        
        plt.savefig("federated_learning_performance_resnet_50_rounds.png")
        print("Plot saved to federated_learning_performance_resnet_50_rounds.png")
        
        plt.show()
    else:
        print("No distributed metrics found in history to plot.")

