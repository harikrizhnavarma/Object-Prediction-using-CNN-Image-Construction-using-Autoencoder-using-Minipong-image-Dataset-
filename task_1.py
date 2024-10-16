import numpy as np
from preparation import DataSet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == '__main__':

    dataset = DataSet()

    # save the datasets into variable
    xTrainData, xTestData, yTrainData, yTestData, zTrainData, zTestData = dataset.dataPreProcess()

    # define the hyper parameters

    epochs = 150
    learningRate = 0.001
    batch_size = 32

    # define the CNN
    class ConvoNet(nn.Module):
        def __init__(self):
            super(ConvoNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 6, kernel_size= 5, padding = 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, padding = 2)

            # for x 
            self.fc1 = nn.Linear(16 * 3 * 3, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 13)

            # for y labels
            self.fc1_y = nn.Linear(16 * 3 * 3, 150)
            self.fc2_y = nn.Linear(150, 95)
            self.fc3_y = nn.Linear(95, 13)

            # for z labels
            self.fc1_z = nn.Linear(16 * 3 * 3, 150)
            self.fc2_z = nn.Linear(150, 95)
            self.fc3_z = nn.Linear(95, 5)

        def forward(self, out):
            out = self.pool(F.relu(self.conv1(out)))
            out = self.pool(F.relu(self.conv2(out)))

            #flatten the images
            out = out.view(-1, 16 * 3 * 3)

            # for x label
            x = F.relu(self.fc1(out))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            # for y
            y = F.relu(self.fc1_y(out))
            y = F.relu(self.fc2_y(y))
            y = self.fc3_y(y)

            # for z labels
            z = F.relu(self.fc1_z(out))
            z = F.relu(self.fc2_z(z))
            z = self.fc3_z(z)

            return x, y, z

    # create the network object
    CNN = ConvoNet()
    CNN.train()


    # defining a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CNN.parameters(), lr = learningRate, momentum = 0.9)

    # train the network
    for epoch in range(epochs):

        x_runningLoss = 0.0
        y_runningLoss = 0.0
        z_runningLoss = 0.0
        for index, (x_data, y_data, z_data) in enumerate(zip(xTrainData, yTrainData, zTrainData), 0):

            x_image, x_label = x_data
            y_image, y_label = y_data
            z_image, z_label = z_data

            #zeroing the parameter gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            x_out, _ , _ = CNN(x_image) # forward passing
            _, y_out , _ = CNN(y_image) # forward passing
            _ , _ , z_out = CNN(z_image) # forward passing


            x_loss = criterion(x_out, x_label) # computing loss

            y_loss = criterion(y_out, y_label) # computing loss

            z_loss = criterion(z_out, z_label) # computing loss

            totalLoss = x_loss + y_loss + z_loss
            totalLoss.backward()

            optimizer.step()

            x_runningLoss += x_loss.item()
            y_runningLoss += y_loss.item()
            z_runningLoss += z_loss.item()
            if index % 32 == 0: # printing for every mini batches
                print('[%d, %5d] x_loss: %.3f, y_loss: %.3f, z_loss: %.3f' %
                (epoch + 1, index + 1, x_runningLoss / 32, y_runningLoss / 32, z_runningLoss / 32))
                x_runningLoss = 0.0
                y_runningLoss = 0.0
                z_runningLoss = 0.0
    print("Training Done !")

    CNN.eval() # set the network to evaluation mode

    correct_x = 0
    correct_y = 0
    correct_z = 0
    total_x = 0
    total_y = 0
    total_z = 0
    with torch.no_grad():

        # for each batch
        for index, (x_data, y_data, z_data) in enumerate(zip(xTestData, yTestData, zTestData), 0):

            x_image, x_label = x_data
            y_image, y_label = y_data
            z_image, z_label = z_data

            x_out, _ , _ = CNN(x_image) # forward passing
            _, y_out , _ = CNN(y_image) # forward passing
            _ , _ , z_out = CNN(z_image) # predicting the probability of each class

            _, x_predicted = torch.max(x_out, 1) # choose the class with max probability
            _, x_trueClass = torch.max(x_label, 1)

            _, y_predicted = torch.max(y_out, 1) # choose the class with max probability
            _, y_trueClass = torch.max(y_label, 1)

            _, z_predicted = torch.max(z_out, 1) # choose the class with max probability
            _, z_trueClass = torch.max(z_label, 1)
            
            total_x += x_label.size(0)
            correct_x += (x_predicted == x_trueClass).sum().item()

            total_y += y_label.size(0)
            correct_y += (y_predicted == y_trueClass).sum().item()

            total_z += z_label.size(0)
            correct_z += (z_predicted == z_trueClass).sum().item()

            # Print actual and predicted values for each image
            print(f"Batch {index + 1}")
            print(f"Actual x: {x_trueClass.tolist()},\n Predicted x: {x_predicted.tolist()}")
            print(f"Actual y: {y_trueClass.tolist()},\n Predicted y: {y_predicted.tolist()}")
            print(f"Actual z: {z_trueClass.tolist()}, \n Predicted z: {z_predicted.tolist()}")

    print("Accuracy for x: {:.2f}%".format(100 * correct_x / total_x))
    print("Accuracy for y: {:.2f}%".format(100 * correct_y / total_y))
    print("Accuracy for z: {:.2f}%".format(100 * correct_z / total_z))