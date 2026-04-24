# Mini Transformer from Scratch

This project implements a small Transformer-based language model inspired by the Transformer architecture.

## Features

* Character-level tokenizer
* Mini Transformer model
* Training loop
* Text generation

## Project Structure

* `src/` → core code
  * `model.py` → Transformer model definition
  * `tokenizer.py` → character-level tokenizer
  * `train.py` → training loop
  * `generate.py` → text generation
  * `utils.py` → helper functions (batching)
* `data/data.txt` → training dataset
* `outputs/checkpoints/` → saved model checkpoints

## How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
cd src
python train.py
```

The checkpoint is saved to `outputs/checkpoints/model.pt`.

### Generate text

```
cd src
python generate.py
```

## Goal

This project is for learning how models like GPT work internally.
