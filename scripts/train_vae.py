from experiments import elbo_loss, vanillavae as vae

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# TODO: Need to refactor the directory structure but I'm tired

# TODO: Obviously do something with these dangling monstrosities
BSZ = 128
EPOCHS = 10

device = ("cuda" if torch.cuda.is_available() else "cpu")

training_data = datasets.CelebA(
    root="data",
    split="train",
    transform=ToTensor(),
    download=True
)

test_data = datasets.CelebA(
    root="data",
    split="test",
    transform=ToTensor(),
    download=True
)

# train_data = DataLoader(training_data, batch_size=BSZ, shuffle=True)

# # TODO: make this nice and pretty with wandb
# def train(model, train_loader, optimizer, device):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         criterion = elbo_loss
#         loss = criterion(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()

#     return train_loss / len(train_loader.dataset)

# model = vae().to(device)
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=1e-3,
#     betas=(0.9, 0.999)
# )

# for epoch in range(EPOCHS):
#     train_loss = train(model, training_data, optimizer, device)
#     print(f'Epoch: {epoch}, Loss: {train_loss:.4f}')
