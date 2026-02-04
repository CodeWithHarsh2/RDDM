import os
import random
import torch
from torchvision import transforms
from PIL import Image
from models.unet import SimpleUNet
from utils.diffusion import make_schedules, forward_diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 10
alpha, beta, alpha_bar, beta_bar = make_schedules(T)

model = SimpleUNet().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


input_dir = os.path.join(os.getcwd(), "dataset_nir", "input")
target_dir = os.path.join(os.getcwd(), "dataset_nir", "target")

files = os.listdir(input_dir)

best_loss = float("inf")

for epoch in range(120):
    random.shuffle(files)

    for file in files:
        I_in = transform(Image.open(os.path.join(input_dir, file))).to(device)
        I_0 = transform(Image.open(os.path.join(target_dir, file))).to(device)

        I_in = I_in.unsqueeze(0)
        I_0 = I_0.unsqueeze(0)

        I_res = I_in - I_0

        t = torch.randint(0, T, (1,)).item()
        I_t, _ = forward_diffusion(I_0, I_res, t, alpha_bar, beta_bar)

        model_input = torch.cat([I_t, I_in], dim=1)
        pred_res = model(model_input)

        loss = torch.mean(torch.abs(pred_res - I_res))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss {loss.item()}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "model.pth")
