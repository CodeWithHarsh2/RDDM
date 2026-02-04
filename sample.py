import torch
from torchvision import transforms
from PIL import Image
from models.unet import SimpleUNet
from utils.diffusion import make_schedules

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 10
_, _, alpha_bar, beta_bar = make_schedules(T)

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

I_in = transform(Image.open("test.png")).unsqueeze(0).to(device)

# Forward diffusion
I = I_in + beta_bar[-1] * torch.randn_like(I_in)

for t in reversed(range(1, T)):
    print("I shape:", I.shape)
    print("I_in shape:", I_in.shape)

    model_input = torch.cat([I, I_in], dim=1)  # [1,2,H,W]

    pred_res = model(model_input)

    # 🔴 CRITICAL FIX
    if pred_res.shape[1] == 3:
        pred_res = pred_res.mean(dim=1, keepdim=True)

    pred_eps = (I - I_in - (alpha_bar[t] - 1) * pred_res) / beta_bar[t]

    I = (
        I
        - (alpha_bar[t] - alpha_bar[t - 1]) * pred_res
        - (beta_bar[t] - beta_bar[t - 1]) * pred_eps
    )

# Save grayscale output
out = (I.clamp(-1, 1) + 1) / 2
out_img = (out[0, 0].detach().cpu().numpy() * 255).astype("uint8")
Image.fromarray(out_img, mode="L").save("output.png")
