# app_fgsm.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
from fgsm import FGSM
from utils import pil_to_tensor, tensor_to_pil, pil_to_base64, softmax_predict
from train_mnist import SimpleCNN  # reuse model class
import base64

app = FastAPI(title="FGSM Attack API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = "mnist_cnn.pth"
DEVICE = "cpu"  # change to "cuda" if available and set appropriate checks

def load_model(path=MODEL_PATH, device=DEVICE):
    model = SimpleCNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()
attacker = FGSM(model, device=DEVICE)

@app.post("/attack")
async def attack(file: UploadFile = File(...), epsilon: float = Form(0.1)):
    # Validate epsilon
    if epsilon < 0 or epsilon > 1.0:
        raise HTTPException(status_code=400, detail="epsilon must be in [0, 1]")

    # Read image bytes
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess to MNIST tensor [1,1,28,28]
    x = pil_to_tensor(pil_img)

    # Make a dummy label prediction for original image to be used for attack.
    clean_pred, clean_probs = softmax_predict(model, x, device=DEVICE)
    # For targeted attack we'd specify y_target; FGSM here uses true label for loss.
    # Since we don't have ground-truth, we can use the model's predicted label as the "true" label
    y = torch.tensor([clean_pred], dtype=torch.long)

    # Generate adversarial example
    x_adv = attacker.perturb(x, y, epsilon=epsilon)

    # Predictions on adversarial
    adv_pred, adv_probs = softmax_predict(model, x_adv, device=DEVICE)

    # Convert adv image to base64
    adv_pil = tensor_to_pil(x_adv)
    adv_b64 = pil_to_base64(adv_pil, fmt='PNG')

    attack_success = (adv_pred != clean_pred)

    result = {
        "clean_prediction": int(clean_pred),
        "clean_probs": clean_probs,          # optional: send probability vector
        "adversarial_prediction": int(adv_pred),
        "adversarial_probs": adv_probs,
        "attack_success": bool(attack_success),
        "epsilon": float(epsilon),
        "adversarial_image_base64": adv_b64
    }
    return JSONResponse(content=result)

# To run use: uvicorn app_fgsm:app --reload --port 8000