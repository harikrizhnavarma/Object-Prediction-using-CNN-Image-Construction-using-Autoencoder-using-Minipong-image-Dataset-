from preparation import DataSet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    dataset = DataSet()
    allData = dataset.allDataPreProcess()

    # define hyper parameters
    epochs = 500
    learningRate = 0.001
    batch_size = 32

    class ConvAutoEncoder(nn.Module):
        def __init__(self):
            super(ConvAutoEncoder, self).__init__()

            # Encoder
            self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1)
            self.conv2 = nn.Conv2d(16, 8, kernel_size = 3, stride = 2, padding = 1)
 

            # Decoder
            self.iconv1 = nn.ConvTranspose2d(8, 16, kernel_size = 3, stride = 2, padding=1, output_padding = 1)
            self.iconv2 = nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 2, padding = 1)

        def encoder(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x

        def decoder(self, x):
            x = F.relu(self.iconv1(x))
            x = torch.tanh(self.iconv2(x))
            return x

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
    
    model = ConvAutoEncoder()
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 0.0001)

    for epoch in range(epochs):

        runningLoss = 0.0

        for index, data in enumerate(allData,0):
            
            image = data[0]

            optimizer.zero_grad()

            #forward pass
            output = model(image)

            #backward
            loss = criterion(output, image)
            loss.backward()

            #optimize
            optimizer.step()

            runningLoss += loss.item()

            if index % batch_size == 0:
                print(f"Loss at epoch:{epoch + 1} = {round(runningLoss, 4)}")

        if epoch + 1  in [100, 200, 300, 400, 500]:

            fig, axes = plt.subplots(1, 2, figsize = (10,5))

            output_img = output[0].detach().squeeze(0).squeeze(0).numpy()

            axes[0].imshow(output_img, cmap = 'coolwarm')
            axes[0].set_title(f"Constructed Image at {epoch + 1} epoch")
            axes[0].axis("off")

            actual_image = image[0].squeeze(0).squeeze(0).numpy()

            axes[1].imshow(actual_image, cmap = 'coolwarm')
            axes[1].set_title(f"Actual Image at {epoch + 1} epoch")
            axes[1].axis("off")
            plt.show()

            

