# CricShot-Action-Recognition
CNN LSTM architecture implemented in Pytorch for Action Recognition that classifies Cricket shots from Video Clips

## Application

This model classifies different cricket shots from a given video. The model is trained on [CricShot10k](https://github.com/Dihan69/CricShot10k)  , a large-scale video dataset for cricket shot classification, consisting of 10,086 video clips across 15 shot classes.

---
## Model Architecture

The network processes raw video sequences through four distinct stages:
1.  **Spatial Encoding (CNN):** Extracts visual features from individual frames independently.
2.  **Temporal Modeling (Bi-LSTM):** Captures the sequential dynamics and motion across the frames.
3.  **Temporal Attention:** Dynamically weighs the importance of each frame in the sequence
4.  **Classification Head:** Maps the final attended context vector to class probabilities.

---

## Input Specification & Preprocessing

Before entering the model, the raw video must be processed into a standardized tensor.

* **Frame Sampling:** Extract exactly $T = 16$ frames per video clip.
* **Resolution:** Resize all frames to $112 \times 112$ pixels.
* **Normalization:** Standard ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
* **Input Tensor Shape:** `(B, T, C, H, W)` $\rightarrow$ `(B, 16, 3, 112, 112)` 
    * *(B = Batch Size)*

---

## Spatial Encoder (CNN)

To process frames efficiently, the input tensor is reshaped from `(B, T, C, H, W)` to `(B * T, C, H, W)`. This allows the CNN to treat every frame in the batch as an independent image, enabling parallel computation.

| Layer Type              | Parameters / Kernel      | Padding | Activation | Output Shape `(B*T, C, H, W)` |
| :---------------------- | :----------------------- | :------ | :--------- | :---------------------------- |
| **Input Reshape**       | N/A                      | N/A     | N/A        | `(B*16, 3, 112, 112)`         |
| **Conv2d (Block 1)**    | 16 filters, $3 \times 3$ | 1       | ReLU       | `(B*16, 16, 112, 112)`        |
| MaxPool2d               | $2 \times 2$, stride 2   | 0       | -          | `(B*16, 16, 56, 56)`          |
| **Conv2d (Block 2)**    | 32 filters, $3 \times 3$ | 1       | ReLU       | `(B*16, 32, 56, 56)`          |
| MaxPool2d               | $2 \times 2$, stride 2   | 0       | -          | `(B*16, 32, 28, 28)`          |
| **Conv2d (Block 3)**    | 64 filters, $3 \times 3$ | 1       | ReLU       | `(B*16, 64, 28, 28)`          |
| **AdaptiveAvgPool2d**   | Output size $(1, 1)$     | -       | -          | `(B*16, 64, 1, 1)`            |
| **Flatten**             | Dimensions $1$ to $-1$   | -       | -          | `(B*16, 64)`                  |
| **Linear (Projection)** | In: 64, Out: 128         | -       | -          | `(B*16, 128)`                 |

**Intermediate Reshape:** Before passing to the LSTM, the tensor is reshaped back to its sequential format: `(B, 16, 128)`.

---

## Temporal Encoder (Bidirectional LSTM)
The sequence of 128-dimensional spatial feature vectors is fed into a recurrent layer to map motion over time.

* **Type:** Bi-Directional Long Short-Term Memory (Bi-LSTM).
* **Input Size:** 128 (matches CNN output).
* **Hidden Size:** 128.
* **Number of Layers:** 1-2
* **Batch First:** `True` (expects inputs as `(Batch, Sequence, Feature)`).
* **Output Shape:** `(B, T, Hidden_Size * 2)` $\rightarrow$ `(B, 16, 256)`.

---

##  Attention Mechanism
Instead of simply taking the final hidden state of the LSTM or averaging them, this module calculates a learned weight for each of the 16 frames.

1.  **Scoring:** A Linear layer maps the 256-dimensional hidden state of each frame to a single scalar score.
    $$e_t = W_a h_t + b_a$$
2.  **Weighting:** A Softmax function normalizes these scores across the time dimension ($T$) so they sum to 1.
    $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$
3.  **Context Vector:** The final representation is the weighted sum of all hidden states.
    $$C = \sum_{t=1}^{T} \alpha_t h_t$$

* **Tensor Shape Transformation:** `(B, 16, 256)` $\rightarrow$ Attention $\rightarrow$ `(B, 256)`.

---

## Classification Head
The context vector is passed through a multi-layer perceptron to generate final class predictions.

| Layer Type          | Parameters                  | Activation        | Output Shape       |
| :------------------ | :-------------------------- | :---------------- | :----------------- |
| **Input (Context)** | from Attention              | -                 | `(B, 256)`         |
| **Linear**          | In: 256, Out: 128           | ReLU              | `(B, 128)`         |
| **Linear (Output)** | In: 128, Out: `num_classes` | (Handled by Loss) | `(B, num_classes)` |

---

## Training & Optimization Configuration

* **Loss Function:** `nn.CrossEntropyLoss()`
* **Optimizer:** `torch.optim.Adam`
* **Learning Rate:** Initial set to `1e-4` ($0.0001$).
* **Learning Rate Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau`
* **Batch Size:** 8
