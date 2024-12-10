import torch
import torchvision.transforms as transforms 
from dermnet_dataset import DermNet
from neural_network import Net
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

if __name__ == "__main__": # Prevents several instances of this script from being ran simultaneously
    # Transform PILIImages (Torchvision output) into Tensors of normalized range
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        # Mean and standard deviation (Std) used for normalization, suited for skin tone RGB values
        transforms.Normalize((0.65, 0.58, 0.50), (0.20, 0.22, 0.23))] 
    )
    trainset = DermNet()
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
    classes = ('Acne', 'Atopic Dermatitis', 'Bacterial Infection', 'Bengin Tumor', 'Bullous Disease',
            'Eczema', 'Lupus', 'Lyme Disease', 'Malignant Lesions (Cancer)', 'Mole', 'Nail Fungus', 'Poison Ivy',
            'STD', 'Viral Infection')

    # Define a neural network
    net = Net()

    # Define optimizer and loss function (criterion object)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Default learning rate and momentum

    # Initialization for plotting
    y_loss_axis = []
    x_epoch_axis = []
    plt.xlabel('Epochs')
    plt.ylabel('Classfication Loss')
    plt.title('Figure 1: Measuring progression of classification loss over a number of epochs')

    # Training
    epochs = int(input("Train for epochs: "))
    for epoch in range(int(epochs)):
        running_loss = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad() # Zero gradient params
            outputs = net(inputs) # Forward pass
            loss = criterion(outputs, labels) # Computes loss
            # Backpropagation to compute the gradients of the loss with repect to all params (how much each contributed)
            loss.backward() 
            optimizer.step() # Optimize 
            
            running_loss += loss.item()
        normalized_running_loss = round(running_loss/len(train_dataloader), 3)
        x_epoch_axis.append(i+1)
        y_loss_axis.append(running_loss)
        print(f'Epoch {epoch+1} loss: {running_loss}')
        
    plt.plot(x_epoch_axis, y_loss_axis, marker='o', linestyle='-', color='b')
    plt.savefig(f'Classifcation_loss_over_{epochs}_epochs.png')
    plt.show()
