# 🖼️ Image Captioning with Deep Learning

This project generates captions for images using a deep learning model built with **CNN (Xception)** for feature extraction and **LSTM** for sequence generation. The application includes a **Streamlit-based UI** and a **command-line test script** with multiple captioning strategies.

---

## 🚀 Features

- Generate image captions using:
  - ✅ Greedy Search
  - ✅ Beam Search
  - ✅ Nucleus Sampling
- Translate captions to:
  - English, Hindi, Marathi, Spanish
- Text-to-Speech (TTS) using `gTTS`
- Download captions as `.txt` or `.csv`
- Streamlit Web App UI

---

## 🧠 Model Architecture

- **Encoder**: Pretrained `Xception` model (without top) to extract 2048D image features.
- **Decoder**: Embedding + LSTM + Dense layers.
- Trained using the **Flickr8k** dataset.

---

First run main.py,train the model,epochs can be changed according to need.
then run test.py,and app.py to run it locally using streamlit.

