# Potato Disease Classification

A Deep Learning project designed to classify potato leaf images into three distinct categories: **Early Blight**, **Late Blight**, and **Healthy**. This application utilizes a Convolutional Neural Network (CNN) built with TensorFlow/Keras and exposes a high-performance REST API built using FastAPI.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Details](#dataset-details)
3. [Model Architecture](#model-architecture)
4. [Training Performance](#training-performance)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Running the Application](#running-the-application)
8. [API Documentation](#api-documentation)
9. [TensorFlow Serving Integration](#tensorflow-serving-integration)

---

## Project Overview

Potato plants are highly susceptible to diseases like **Early Blight** (caused by *Alternaria solani*) and **Late Blight** (caused by *Phytophthora infestans*). Early detection of these diseases can prevent severe crop damage and financial losses for farmers.

This project provides an end-to-end Machine Learning pipeline:
* **Model Training:** CNN classifier trained on preprocessed leaf images.
* **API Deployment:** FastAPI web backend that takes image uploads and returns classification results with confidence scores.
* **Model Serving:** Optional configuration for TensorFlow Serving to serve models in production environments.

---

## Dataset Details

The model is trained on the popular **PlantVillage Dataset**, specifically the potato leaf section.

* **Classes:**
  1. `Early Blight` (Diseased)
  2. `Late Blight` (Diseased)
  3. `Healthy` (No Disease)
* **Image Properties:**
  * Dimensions: `256 x 256` pixels
  * Channels: `3` (RGB)
* **Dataset Splits:**
  * **Train Split:** 80% (used for training parameters)
  * **Validation Split:** 10% (used for hyperparameter tuning & monitoring)
  * **Test Split:** 10% (used for final unseen evaluation)

---

## Model Architecture

The neural network is a custom **Convolutional Neural Network (CNN)** implemented in TensorFlow/Keras.

### Preprocessing & Data Augmentation Layers
1. **Resizing:** Standardizes all input images to `256 x 256` dimensions.
2. **Rescaling:** Normalizes pixel values from `[0, 255]` to `[0, 1]` for stable training gradients.
3. **Data Augmentation** *(Defined in notebook)*: Prepares the network to handle variance by randomly applying horizontal/vertical flips and rotations.

### Network Architecture
The core classification network consists of 6 convolutional layers, each followed by max-pooling:

| Layer (Type) | Filter Config / Activation | Output Shape / Pooling |
| :--- | :--- | :--- |
| **Input** | - | `(256, 256, 3)` |
| **Resizing & Rescaling** | Normalization | `(256, 256, 3)` |
| **Conv2D + MaxPool** | 32 filters (3x3), Relu | `(127, 127, 32)` |
| **Conv2D + MaxPool** | 64 filters (3x3), Relu | `(60, 60, 64)` |
| **Conv2D + MaxPool** | 64 filters (3x3), Relu | `(28, 28, 64)` |
| **Conv2D + MaxPool** | 64 filters (3x3), Relu | `(12, 12, 64)` |
| **Conv2D + MaxPool** | 64 filters (3x3), Relu | `(4, 4, 64)` |
| **Conv2D + MaxPool** | 64 filters (3x3), Relu | `(1, 1, 64)` |
| **Flatten** | - | `64` |
| **Dense** | 64 units, Relu | `64` |
| **Dense (Output)** | 3 units, Softmax | `3` (Class probabilities) |

---

## Training Performance

The model was trained for **50 Epochs** using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss.

* **Final Training Accuracy:** ~100.0%
* **Final Validation Accuracy:** ~99.48%
* **Final Validation Loss:** `0.0045`
* **Test Dataset Accuracy:** 100.0% (Loss: `2.19e-05`)

---

## Project Structure

```
potato-disease-classification/
├── PlantVillage/               # Dataset directory containing subfolders:
│   ├── Potato___Early_blight/  # Early blight leaf images
│   ├── Potato___Late_blight/   # Late blight leaf images
│   └── Potato___healthy/       # Healthy leaf images
├── api/                        # FastAPI application files
│   ├── main.py                 # Core API code (Keras model loading)
│   ├── main-tf-serving.py      # Alternative API code optimized for TF serving
│   └── requirements.txt        # API-specific dependencies
├── saved_models/               # Pretrained models folder
│   ├── 1.h5                    # Production Model (v1)
│   └── 2.h5                    # Beta/Alternative Model (v2)
├── training/                   # Model development environment
│   └── training.ipynb          # Jupyter notebook for training and evaluation
├── main.py                     # Root entry script
├── models.config               # Config file for TF Serving server routing
├── pyproject.toml              # Project metadata & standard python configuration
├── requirements.txt            # Root dependencies (TensorFlow & Matplotlib)
└── README.md                   # Project documentation
```

---

## Installation & Setup

This project uses Python `>=3.13` and can be set up using either `uv` (recommended) or standard `pip`.

### 1. Clone the repository
```bash
git clone <repository_url>
cd potato-disease-classification
```

### 2. Set up Virtual Environment & Dependencies
**Using `uv` (recommended):**
```bash
# Sync dependencies and activate environment
uv sync
```

**Using standard `pip`:**
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r api/requirements.txt
```

---

## Running the Application

### 1. Run the FastAPI Server
Launch the application server:
```bash
cd api
uvicorn main:app --reload --port 8000
```
The server will start at `http://localhost:8000`. You can visit the API documentation page at `http://localhost:8000/docs`.

> [!NOTE]
> If you run into model loading errors, make sure the path in [api/main.py](file:///e:/Technical/potato-disease-classification/api/main.py) points to the correct model format (e.g. changing `../saved_models/1` to `../saved_models/1.h5`).

### 2. Run TF Serving Backend (Docker)
To run TensorFlow Serving to host your saved models, start the Docker container mapping your model path:
```bash
docker run -t --rm -p 8501:8501 \
    -v "/path/to/potato-disease-classification:/workspace" \
    tensorflow/serving \
    --model_config_file=/workspace/models.config
```

---

## API Documentation

### 1. Health Check
* **Endpoint:** `GET /ping`
* **Response:**
```json
"Hello, I am alive"
```

### 2. Leaf Disease Prediction
* **Endpoint:** `POST /predict`
* **Payload:** `multipart/form-data` with key `file` (Image file)
* **Response:**
```json
{
  "class": "Early Blight",
  "confidence": 0.9854
}
```

---

## TensorFlow Serving Integration

For production grade serving, configure the models via the provided `models.config` file. It mounts and exposes the `potatoes_model` and routes requests automatically based on model version settings.

```protobuf
model_config_list {
 config {
    name: 'potatoes_model'
    base_path: '/workspace/saved_models'
    model_platform: 'tensorflow'
    model_version_policy: {all: {}}
        }
}
```
*(Make sure to adjust the host configurations in `models.config` to match your local paths/volume mounting points!)*
