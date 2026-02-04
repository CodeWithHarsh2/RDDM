import torch
from torchvision import transforms
from PIL import Image
from models.unet import SimpleUNet
from utils.diffusion import make_schedules

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 10
_, _, alpha_bar, beta_bar = make_schedules(T)

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

I_in = transform(Image.open("test.png")).unsqueeze(0).to(device)

I = I_in + beta_bar[-1] * torch.randn_like(I_in)

for t in reversed(range(1, T)):
    model_input = torch.cat([I, I_in], dim=1)
    pred_res = model(model_input)

    pred_eps = (I - I_in - (alpha_bar[t] - 1) * pred_res) / beta_bar[t]

    I = I - (alpha_bar[t] - alpha_bar[t-1]) * pred_res \
          - (beta_bar[t] - beta_bar[t-1]) * pred_eps

out = (I.clamp(-1,1) + 1) / 2
Image.fromarray((out[0].detach().permute(1,2,0).cpu().numpy()*255).astype("uint8")).save("output.png")

