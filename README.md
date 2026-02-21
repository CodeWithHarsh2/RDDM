# RDDM

RDDM is a Python implementation of a diffusion based deep learning model for image generation and experimentation.  
The project includes training, sampling, dataset handling, and utility modules in a clean and modular structure.

## Research References

This work is based on the following two papers:

1. **Residual Denoising Diffusion Models** — a dual diffusion framework that decouples denoising into residual diffusion and noise diffusion, unifying image generation and restoration tasks. https://arxiv.org/abs/2308.13712
2. **Deep Learning for In Vivo Near-Infrared Imaging** — explores deep learning enhancement of near-infrared (NIR) imaging quality, which is used here for dataset evaluation and comparison. https://pmc.ncbi.nlm.nih.gov/articles/PMC7817119/


## Project Structure

```
RDDM/
│── dataset/            # Dataset loading and preprocessing
│── dataset_nir/        # Additional dataset
│── models/             # Model definitions
│── utils/              # Helper utilities
│── train.py            # Training script
│── sample.py           # Sampling / inference script
│── requirements.txt    # Project dependencies
│── test.png            # Example input
│── output.png          # Example output
```

## Setup

Clone the repository:

```
git clone https://github.com/CodeWithHarsh2/RDDM.git
cd RDDM
```

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Train the model:

```
python train.py
```

Generate samples:

```
python sample.py
```

## Notes

- Place your dataset inside the appropriate dataset folder before training.
- Modify hyperparameters directly inside the training script if needed.


