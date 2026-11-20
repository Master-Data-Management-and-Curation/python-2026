# Session 3 - BentoML for image classification

## Today's goal

Set up an image recognition service for SEM images using BentoML.

Add the labeled category to the payload.

## Start

```bash
ssh <your-vm>
cd python-2026/py4DM-2 && source .venv/bin/activate
pip install -e .
docker compose up -d                
python verify.py                    # lists sem_images among the collections
```

If `sem_images` is missing, rebuild it: `python s1/solution/build_collection.py` (5 seconds).

## 1. `service.py`

1.1 apply image pre-processing to get PyTorch tensors

1.2 pass the input to the model and apply softmax 

1.3 pass from the number to the label

1.4 what classifier we are using (loaded once)

1.5 what we want as output

1.6 `bentoml serve`

1.7 test with an image

## 2. `classify.py`

2.1 what we want to write in the payload​

2.2 connection to the service​

2.3 check if the predicted category matches the true one (for accuracy)​
