!pip install -q datasets pillow

from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs("/content/train_images", exist_ok=True)
os.makedirs("/content/nonmember_images", exist_ok=True)

# =========================================================
# LOAD CELEBA
# =========================================================

dataset = load_dataset("tpremoli/CelebA-attrs", split="train")

print("Dataset loaded:", len(dataset))

# =========================================================
# TAKE SMALL SUBSET FOR T4
# =========================================================

member_count = 50
nonmember_count = 50

# =========================================================
# SAVE MEMBER IMAGES
# =========================================================

print("Saving member images...")

for i in tqdm(range(member_count)):

    img = dataset[i]["image"]

    img.save(f"/content/train_images/{i}.jpg")

# =========================================================
# SAVE NON-MEMBER IMAGES
# =========================================================

print("Saving non-member images...")

offset = 500

for i in tqdm(range(nonmember_count)):

    img = dataset[offset + i]["image"]

    img.save(f"/content/nonmember_images/{i}.jpg")

print("\nDONE!")

print("Train images:",
      len(os.listdir('/content/train_images')))

print("Non-member images:",
      len(os.listdir('/content/nonmember_images')))

# ============================================================
# BLACK-BOX MEMBERSHIP INFERENCE ATTACK
# CLOSE REPRODUCTION OF NDSS 2025 PAPER
# "Black-box Membership Inference Attacks against Fine-tuned Diffusion Models"
# ============================================================

!pip install -q diffusers transformers accelerate datasets ftfy scipy \
bitsandbytes sentencepiece timm scikit-learn torchvision

import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoImageProcessor,
    DeiTModel
)

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler
)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier

from torchvision import transforms

device = "cuda"

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "runwayml/stable-diffusion-v1-5"

TRAIN_DIR = "/content/train_images"
NONMEMBER_DIR = "/content/nonmember_images"

OUTPUT_DIR = "/content/fine_tuned_sd"

NUM_GENERATIONS = 3
IMG_SIZE = 512

# ============================================================
# LOAD SD PIPELINE
# ============================================================

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe.enable_attention_slicing()

# ============================================================
# OPTIONAL: LOAD BLIP CAPTIONER
# ============================================================

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16
).to(device)

# ============================================================
# LOAD DEIT FEATURE EXTRACTOR
# ============================================================

deit_processor = AutoImageProcessor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)

deit_model = DeiTModel.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
).to(device)

deit_model.eval()

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt").to(device)

    out = caption_model.generate(
        **inputs,
        max_new_tokens=30
    )

    caption = caption_processor.decode(
        out[0],
        skip_special_tokens=True
    )

    return caption

def extract_embedding(image):

    inputs = deit_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = deit_model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1)

    return emb.cpu().numpy()

def cosine_score(img1, img2):

    e1 = extract_embedding(img1)
    e2 = extract_embedding(img2)

    return cosine_similarity(e1, e2)[0][0]

def generate_image(prompt):

    with torch.autocast("cuda"):

        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

    return image

# ============================================================
# ATTACK FEATURE EXTRACTION
# ============================================================

def compute_similarity_vector(image_path):

    real_img = Image.open(image_path).convert("RGB")

    # generate caption
    caption = generate_caption(real_img)

    scores = []

    for _ in range(NUM_GENERATIONS):

        fake_img = generate_image(caption)

        score = cosine_score(real_img, fake_img)

        scores.append(score)

    return np.array(scores)

# ============================================================
# BUILD DATASET
# ============================================================

member_paths = [
    os.path.join(TRAIN_DIR, x)
    for x in os.listdir(TRAIN_DIR)
]

nonmember_paths = [
    os.path.join(NONMEMBER_DIR, x)
    for x in os.listdir(NONMEMBER_DIR)
]

X = []
y = []

print("Processing MEMBER samples...")

for p in tqdm(member_paths):

    try:
        feat = compute_similarity_vector(p)

        X.append(feat)
        y.append(1)

    except Exception as e:
        print(e)

print("Processing NON-MEMBER samples...")

for p in tqdm(nonmember_paths):

    try:
        feat = compute_similarity_vector(p)

        X.append(feat)
        y.append(0)

    except Exception as e:
        print(e)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# ============================================================
# TRAIN ATTACK MODEL
# ============================================================

clf = MLPClassifier(
    hidden_layer_sizes=(64,32),
    max_iter=300
)

clf.fit(X, y)

pred = clf.predict(X)
prob = clf.predict_proba(X)[:,1]

acc = accuracy_score(y, pred)
auc = roc_auc_score(y, prob)

print("\n==============================")
print("ATTACK RESULTS")
print("==============================")
print("Accuracy:", acc)
print("ROC-AUC:", auc)

# ============================================================
# TEST SINGLE IMAGE
# ============================================================

def infer_membership(image_path):

    feat = compute_similarity_vector(image_path)

    feat = feat.reshape(1,-1)

    prob = clf.predict_proba(feat)[0][1]

    print("\nMembership Probability:", prob)

    if prob > 0.5:
        print("=> MEMBER")
    else:
        print("=> NON-MEMBER")

# Example:
# infer_membership("/content/test.jpg")
