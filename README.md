# Face Recognition System using FaceNet and Inception Model

This project implements a **face recognition system** using **FaceNet** with **Inception blocks** in Keras. It can **verify a person's identity** or **recognize who the person is** by comparing face embeddings with a database of known faces.

---

## Features

- Custom Keras implementation of **FaceNet** with multiple **Inception blocks**.
- Uses **triplet loss** for training embeddings.
- Converts images to **128-dimensional embeddings** for face comparison.
- Face **verification**: checks if a person matches a specific identity.
- Face **recognition**: identifies the closest person in the database.
- Load pre-trained weights exported from **OpenFace Torch model**.

---

## Files

- `main.ipynb` - Main notebook with training, verification, and recognition code.
- `utils.py` - Helper functions:
  - `conv2d_bn()` – convolution + batch normalization + activation block.
  - `img_to_encoding()` – converts image to embedding.
  - `load_weights_from_FaceNet()` – loads pre-trained weights.
- `weights/` - Folder containing CSV files with pre-trained weights.
- `images/` - Sample images for testing.

---

## Acknowledgments

- Concepts and exercises from the Coursera course: [Convolutional Neural Networks by Andrew Ng](https://www.coursera.org/learn/convolutional-neural-networks)

