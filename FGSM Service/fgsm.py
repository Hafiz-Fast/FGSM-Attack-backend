# fgsm.py
import torch
import torch.nn.functional as F

class FGSM:
    """
    FGSM attack:
    Given model, input image tensor x (requires_grad=True), and label y,
    compute adversarial example: x_adv = clamp(x + epsilon * sign(grad_x loss), 0, 1)
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def perturb(self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1):
        """
        x: tensor shape [B, C, H, W], values in [0,1], requires_grad False or True
        y: tensor shape [B] (long)
        returns x_adv tensor of same shape
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        # ensure grads on input
        x_adv = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            loss.backward()

        # sign gradient
        grad_sign = x_adv.grad.data.sign()
        perturbed = x_adv + epsilon * grad_sign
        perturbed = torch.clamp(perturbed, 0.0, 1.0)
        return perturbed.detach()
