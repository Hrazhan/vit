import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader
from model import ViT
from accelerate import Accelerator
from torchinfo import summary

IMAGE_SIZE = 224

transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor()])

train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)


BATCH_SIZE = 256
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_data.classes



model = ViT(image_size=224,
            in_channels=3,
            patch_size=28, 
            embed_dim=384,
            num_transformer_layer=6,
            num_heads=6,
            hidden_units=1028)
# model = torch.compile(model)
random_image = (1, 3, 224, 224)

summary(model, 
        input_size=random_image,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader
)

device = accelerator.device
print(f"Device: {device}")
model.to(device)

EPOCHS = 5

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_loss, train_acc = 0, 0
    for images, labels in train_dataloader:
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        train_loss += loss.item()
        train_acc += torch.argmax(torch.softmax(outputs, dim=1), dim=1).sum().item()
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for images, labels in test_dataloader:

            test_outputs = model(images)
            loss = loss_fn(test_outputs, labels)
            test_loss += loss.item()
            test_acc += torch.argmax(torch.softmax(test_outputs, dim=1), dim=1).sum().item()

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    
    accelerator.print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')