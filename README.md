# Face Recognition System using FaceNet and Inception

This project implements a **Face Recognition System** using the **FaceNet architecture** with **Inception blocks** built in Keras.

The system generates **128-dimensional face embeddings** that can be used to verify a person's identity or recognize individuals by comparing their embeddings with a database of known faces.

---

## Project Overview

The model processes facial images and converts them into numerical representations called **embeddings**.  
These embeddings allow the system to measure similarity between faces using distance metrics.

The system supports two main tasks:

- **Face Verification** – Check if a given image matches a claimed identity.
- **Face Recognition** – Identify the closest matching person from a database.

---

## Features

- Custom **Keras implementation of FaceNet**
- Multiple **Inception blocks** for feature extraction
- **Triplet Loss** used for learning face embeddings
- Converts images into **128-dimensional embeddings**
- Supports both **face verification** and **face recognition**


---
> This project is based on concepts from the **Deep Learning Specialization**  
> https://learn.deeplearning.ai/specializations
