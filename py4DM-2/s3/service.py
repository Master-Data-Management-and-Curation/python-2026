"""Session 3 - BentoML service (exercise): send an SEM image, get its category back.
 
Fill in the TODOs (1.1 - 1.5), then run (from s3/):  bentoml serve 
Test: curl -X POST http://localhost:8081/classify -F image=@some_sem_image.png
 
The classifier is the ViT t0m-R/vit-sem-classification (ViT-B/32 fine-tuned on SEM
images, 10 categories), loaded once at startup.

"""
import bentoml
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
 
MODEL_ID = "t0m-R/vit-sem-classification"
BATCH = 32      # images per forward pass: bounds the memory used when predict gets a long list
 
 
class VitClassifier:
    def __init__(self, model_id: str = MODEL_ID):
        self.processor = AutoImageProcessor.from_pretrained(model_id)  # pre-processing configuration for images
        self.model = AutoModelForImageClassification.from_pretrained(model_id).eval()  # weights and architecture, evaluation mode
        self.id2label = self.model.config.id2label  # from numbers to labels
 
    @torch.inference_mode()
    def predict(self, images) -> list[tuple[str, float]]:
        """[(label, probability)] for a list of PIL images."""
        out = []
        for i in range(0, len(images), BATCH):
            batch = [im.convert("RGB") for im in images[i:i + BATCH]]
            inputs = None  # TODO (1.1) apply the processor to `batch`, asking for PyTorch tensors (images=batch, return_tensors="pt")
            probs = None   # TODO (1.2) run the model on **inputs, take its .logits and apply softmax on the last dimension (**inputs).logits.softmax(-1)
            conf, idx = probs.max(-1)  # selecting maximum confidence result
            out += [(None, float(c)) for c, j in zip(conf, idx)]  # TODO (1.3) replace None with the label of class int(j)
        return out
 
 
@bentoml.service(traffic={"timeout": 60})
class SemClassifier:
    def __init__(self) -> None:
        self.clf = None  # TODO (1.4) What classifier are we using? Which model? (loaded once, at startup)
 
    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        (label, confidence), = self.clf.predict([image])
        return {}  # TODO (1.5) What format and information do we want back? (category, confidence to 3 decimals, model)
