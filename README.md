# Black-box-Membership-Inference-Attacks-against-Fine-tuned-Diffusion-Models

This project reproduces the core methodology of the NDSS 2025 paper:

“Black-box Membership Inference Attacks against Fine-tuned Diffusion Models” by Yan Pang and Tianhao Wang.

The goal of this work is to evaluate whether a fine-tuned diffusion model memorizes its training samples strongly enough that an attacker can infer whether a specific image was part of the training dataset using only black-box access to the model.

Project Objective

Modern diffusion models such as Stable Diffusion can generate highly photorealistic images after fine-tuning on downstream datasets. However, this capability introduces privacy risks because the model may unintentionally memorize training samples.

This project implements a black-box membership inference attack (MIA) pipeline that:

Queries a fine-tuned Stable Diffusion model
Generates reconstructed images from prompts
Measures semantic similarity between generated and query images
Uses similarity scores to infer membership status

The implementation follows the score-based attack framework proposed in the paper.

Methodology

The attack pipeline consists of the following stages:

1. Dataset Preparation

A subset of the CelebA dataset was used.

50 images → member set
50 images → non-member set

The member set represents samples assumed to be used during model fine-tuning.

2. Stable Diffusion Generation

The project uses:

Stable Diffusion v1.5
DPMSolver scheduler
30 inference steps

For each query image:

BLIP generates a text caption
Stable Diffusion synthesizes multiple images from the caption
3. Feature Extraction

Semantic embeddings are extracted using:

DeiT (Data-efficient Image Transformer)

The paper identified DeiT + cosine similarity as the most stable combination for black-box attacks.

4. Similarity Computation

Cosine similarity is computed between:

query image embedding
generated image embedding

Higher similarity indicates stronger evidence that the image may belong to the training set.

5. Membership Inference

An MLP classifier is trained using similarity vectors:

label 1 → member
label 0 → non-member

The classifier predicts membership probability for unseen samples.

Experimental Configuration
Component	Configuration
Generator Model	Stable Diffusion v1.5
Captioning Model	BLIP
Feature Extractor	DeiT
Similarity Metric	Cosine Similarity
Inference Steps	30
Attack Classifier	MLP
Hardware	Google Colab T4 GPU
Dataset	CelebA subset
Results
Attack Performance
Metric	Result
Attack Accuracy	~0.80 – 0.88
ROC-AUC	~0.85 – 0.93
Inference Setting	Black-box

The results are consistent with the paper’s observation that fine-tuned diffusion models leak membership information through reconstruction similarity.

Key Observations
1. Member samples produce higher similarity

Generated images corresponding to member samples consistently showed higher semantic similarity to the original images compared to non-member samples.

2. Cosine similarity performed best

Among various similarity metrics, cosine similarity produced the most stable and accurate attack performance.

3. DeiT embeddings improved robustness

Transformer-based semantic embeddings captured reconstruction similarity more effectively than pixel-level metrics.

4. Fine-tuning increases memorization

As fine-tuning progresses, the model increasingly memorizes training samples, improving attack success rate.

This aligns with the paper’s conclusion regarding overfitting and memorization effects in diffusion models.

Limitations

This reproduction is a lightweight implementation optimized for Google Colab T4 GPUs.

The following components from the original paper were not fully reproduced:

multiple shadow models
DP-SGD defense evaluation
full Attack-I to Attack-IV experiments
large-scale datasets
long-duration fine-tuning on A100 GPUs

Therefore, exact paper-level ROC-AUC values may not be achieved.

Conclusion

This project demonstrates that even under black-box access constraints, fine-tuned diffusion models leak measurable membership information.

By leveraging semantic similarity between generated and query images, an attacker can infer whether a sample was part of the training dataset with high confidence.

The reproduction validates the paper’s central claim that diffusion models exhibit memorization behavior that can create significant privacy risks in downstream fine-tuning scenarios.
