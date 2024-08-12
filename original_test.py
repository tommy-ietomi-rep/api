import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import onnx
import onnxruntime

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor

#define
BATCH_SIZE = 50
LEARNING_RATE = 0.01
EPOCH = 20
MODEL_NAME_ONNX = 'mnist.onnx'

#Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Define the training function
def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        train_losses.append(loss.item())

# Define the test function
def test(model, device, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if True : # re train
    # Create an instance of the model and move it to the device
    model = Net().to(device)

    # Define the optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    # Train and test the model
    train_losses = []
    test_losses = []
    for epoch in range(1, EPOCH):
        train(model, device, train_loader, optimizer, epoch, train_losses)
        test(model, device, test_loader, test_losses)

    # Plot the train and test loss over time
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the model as an ONNX file
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_input, MODEL_NAME_ONNX, input_names=input_names, output_names=output_names)

# Get the first test data image and label
image, label = test_dataset[0]
    
if False : #show data
    # Display the image and label using Matplotlib
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

# Load the model as an ONNX file
ort_session = onnxruntime.InferenceSession(MODEL_NAME_ONNX)

# Run the model on the first test data image
image = image.to(device)
image = image.unsqueeze(0)
input_names = ['input']
outputs = ort_session.run(None, {input_names[0]: image.cpu().numpy()})
predicted_label = torch.argmax(torch.from_numpy(outputs[0]), dim=1).item()

# Display the first test data image and the predicted label
plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
plt.title(f'ONNX runtime Predicted Label: {predicted_label}, True Label: {label}')
plt.show()

# Convert the PyTorch model to a Relay function
input_shape = (1, 1, 28, 28)
input_name = 'input'
output_name = 'output'

# Load the ONNX model
onnx_model = onnx.load(MODEL_NAME_ONNX)
mod, params = relay.frontend.from_onnx(onnx_model, shape={input_name: input_shape})

# Compile the Relay function to a TVM module
target = 'llvm'
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

# Create a TVM runtime module from the compiled module and parameters
module = graph_executor.create(graph, lib, tvm.cpu(0))
module.set_input(input_name, tvm.nd.array(image.numpy().reshape(input_shape).astype('float32')))
module.set_input(**params)

# Run the TVM module on the first test data image
module.run()
tvm_output = module.get_output(0).numpy()
predicted_label = np.argmax(tvm_output)

# Display the first test data image and the predicted label
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f'TVM Predicted Label: {predicted_label}, True Label: {label}')
plt.show()