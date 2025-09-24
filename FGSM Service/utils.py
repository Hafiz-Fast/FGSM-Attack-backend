# utils.py
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
import numpy as np

# For MNIST: grayscale, 28x28
mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()  # yields [0,1]
])

def pil_to_tensor(img_pil):
    """PIL image -> tensor batch [1,1,28,28] in [0,1]"""
    t = mnist_transform(img_pil)
    return t.unsqueeze(0)

def tensor_to_pil(img_tensor):
    """Tensor [1,1,28,28] or [1,28,28] -> PIL Image in mode 'L'"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)
    # img_tensor is [H,W] or [1,H,W]
    arr = (img_tensor.detach().cpu().numpy() * 255.0).astype('uint8')
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr.squeeze(), mode='L')

def pil_to_base64(img_pil, fmt='PNG'):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    b = buf.getvalue()
    return base64.b64encode(b).decode('utf-8')

def softmax_predict(model, x_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.to(device))
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1)
        return pred.cpu().item(), probs.cpu().numpy().tolist()
