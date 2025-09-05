import torch, torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np

class Embedder:
    def __init__(self, device=None):
        weights = ResNet50_Weights.IMAGENET1K_V2
        m = resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(m.children())[:-1]).eval()
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self.model.parameters()).device
        self.tf = weights.transforms()  # includes resize, center crop, normalize

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> np.ndarray:
        x = self.tf(pil_img).unsqueeze(0).to(self.device)  # [1,3,224,224]
        f = self.model(x).squeeze().detach().cpu().numpy() # [2048]
        # L2 normalize for cosine/Annoy angular
        f = f / (np.linalg.norm(f) + 1e-12)
        return f