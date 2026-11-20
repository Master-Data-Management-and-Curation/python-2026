"""Session 3 - BentoML service: send an SEM image, get its category back.
 
Run (from s3/):  bentoml serve
Test: curl -X POST http://localhost:3000/classify -F image=@some_sem_image.png
 
The classifier is the ViT t0m-R/vit-sem-classification (ViT-B/32 fine-tuned on SEM
images, 10 categories), loaded once at startup. No Qdrant and no embedding service
are needed to classify.
 
Needs:  pip install transformers torch
"""
import bentoml
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
 
MODEL_ID = "t0m-R/vit-sem-classification"
BATCH = 32      # images per forward pass: bounds the memory used when predict gets a long list
 
 
class VitClassifier:
    def __init__(self, model_id: str = MODEL_ID):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).eval()
        self.id2label = self.model.config.id2label
 
    @torch.inference_mode()
    def predict(self, images) -> list[tuple[str, float]]:
        """[(label, probability)] for a list of PIL images."""
        out = []
        for i in range(0, len(images), BATCH):
            batch = [im.convert("RGB") for im in images[i:i + BATCH]]
            inputs = self.processor(images=batch, return_tensors="pt")
            probs = self.model(**inputs).logits.softmax(-1)
            conf, idx = probs.max(-1)
            out += [(self.id2label[int(j)], float(c)) for c, j in zip(conf, idx)]
        return out
 
 
@bentoml.service(traffic={"timeout": 60})
class SemClassifier:
    def __init__(self) -> None:
        self.clf = VitClassifier(MODEL_ID)
 
    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        (label, confidence), = self.clf.predict([image])
        return {"category": label, "confidence": round(confidence, 3), "model": MODEL_ID}
