import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.onnx
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("mnist")
mlflow.pytorch.autolog()# Connect to remote MLflow server

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 32 = input size
        # 26 * 26 = conv2d output with kernel 3 becomes 26x26 
        self.fc1 = nn.Linear(32 * 26 * 26, 10) 
    
    def forward(self, x):
        # x.shape [64, 1, 28, 28], only one channel because of black and white
        x = self.conv1(x) # x.shape [64, 32, 26, 26], become 32 channel
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
def main():
    params = {
        "batch_size": 64,
        "learning_rate": 0.001,
    }

    with mlflow.start_run():
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True)

        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        model = MNISTModel().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        print("Starting training..")
        model.train()

        train_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
        
        mlflow.log_metrics({
            "train_loss": train_loss / len(trainloader)
        })

        print("training finished")

        # set to inference mode
        model.eval()

        dummy_input = torch.randn(1,1,28,28, device=device) # batch, channel, height, width
        onnx_path = "mnist.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        mlflow.pytorch.log_model(model, name="mnist")
        print(f"Model exported to {onnx_path}")



if __name__ == "__main__":
    main()
