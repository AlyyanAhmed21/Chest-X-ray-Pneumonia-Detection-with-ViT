---
title: Pneumonia Detection AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.26.0"
app_file: app.py
pinned: false
secrets:
  - MONGODB_CONNECTION_STRING
  # HF_TOKEN is no longer needed here, it's a secret for the runner
---

# 🩺 Pneumonia Detection AI

This Space demonstrates a complete, end-to-end MLOps pipeline for medical image classification.

## ✨ Features

-   **AI-Powered Diagnosis:** Upload one or more chest X-ray images to get an instant classification of **Normal** or **Pneumonia**.
-   **Advanced Model:** Powered by a fine-tuned **Vision Transformer (ViT)** for high accuracy.
-   **Multi-Image Analysis:** The AI provides both an overall prediction for the patient and individual watermarked results for each image.
-   **Patient History:** All analyses are logged to a **MongoDB** database and can be reviewed.
-   **Sample Library:** Test the app instantly with a library of sample X-ray images.

## 🛠️ Tech Stack

-   **Model:** Google's `vit-base-patch16-224-in21k`
-   **MLOps Pipeline:** DVC & MLflow
-   **Frontend:** Gradio
-   **Database:** MongoDB Atlas
-   **Hosting:** Hugging Face Spaces

This project was developed by **Alyyan Ahmed** and **Munim Akbar**.

---
**Disclaimer:** This is a demo application for educational and portfolio purposes. It is **not a certified medical device** and should not be used for actual medical diagnosis.
