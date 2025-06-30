import streamlit as st
import numpy as np
from pickle import load
from PIL import Image
from keras.applications.xception import Xception
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
import io
from gtts import gTTS
import base64
from googletrans import Translator
import csv

@st.cache_resource
def load_tokenizer():
    return load(open("tokenizer.p", "rb"))

tokenizer = load_tokenizer()
vocab_size = len(tokenizer.word_index) + 1
max_length = 32

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

@st.cache_resource
def load_caption_model():
    model = define_model(vocab_size, max_length)
    model_path = "model-13.h5"
    model.load_weights(model_path)
    return model

model = load_caption_model()

@st.cache_resource
def load_xception():
    return Xception(include_top=False, pooling="avg")

xception_model = load_xception()

def extract_features(image, model):
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def beam_search_predictions(model, tokenizer, photo, max_length, beam_index=3):
    start = [tokenizer.word_index['start']]
    sequences = [[start, 0.0]]
    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            sequence = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([photo, sequence], verbose=0)[0]
            top_preds = np.argsort(preds)[-beam_index:]
            for word in top_preds:
                next_seq = seq + [word]
                next_score = score - np.log(preds[word] + 1e-9)
                all_candidates.append([next_seq, next_score])
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_index]
    final_captions = []
    for seq, score in sequences:
        caption = []
        for idx in seq:
            word = word_for_id(idx, tokenizer)
            if word is None or word == 'end':
                break
            if word != 'start':
                caption.append(word)
        final_captions.append(' '.join(caption))
    return final_captions[:3]

def nucleus_sampling_predictions(model, tokenizer, photo, max_length, p=0.9, num_captions=3):
    captions = []
    for _ in range(num_captions):
        in_text = ['start']
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([' '.join(in_text)])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            preds = model.predict([photo, sequence], verbose=0)[0]
            sorted_indices = np.argsort(preds)[::-1]
            cumulative_probs = np.cumsum(preds[sorted_indices])
            cutoff = np.where(cumulative_probs > p)[0][0]
            candidates = sorted_indices[:cutoff+1]
            next_word = np.random.choice(candidates, p=preds[candidates]/preds[candidates].sum())
            word = word_for_id(next_word, tokenizer)
            if word is None or word == 'end':
                break
            in_text.append(word)
        captions.append(' '.join(in_text[1:]))
    return captions

# Streamlit UI
st.title("Image Captioning Generator")
st.write("Upload an image and generate captions using a deep learning model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
languages = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr', 'Spanish': 'es'}
language = st.selectbox("Select language for caption:", list(languages.keys()))
p_value = st.slider("Nucleus Sampling p-value (diversity)", min_value=0.7, max_value=1.0, value=0.9, step=0.01)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Generating captions...")
        photo = extract_features(img, xception_model)
        captions = nucleus_sampling_predictions(model, tokenizer, photo, max_length, p=p_value, num_captions=3)
        st.subheader("Top 3 Captions:")
        translator = Translator()
        translated_captions = []
        for i, cap in enumerate(captions):
            if language != 'English':
                translated = translator.translate(cap, dest=languages[language]).text
            else:
                translated = cap
            translated_captions.append(translated)
            st.success(f"{i+1}. {translated}")
            # Voice narration
            tts = gTTS(translated, lang=languages[language])
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3', start_time=0)
        # Download as txt
        txt_content = '\n'.join(translated_captions)
        st.download_button("Download Captions as TXT", txt_content, file_name="captions.txt")
        # Download as csv
        csv_fp = io.StringIO()
        writer = csv.writer(csv_fp)
        writer.writerow(["Caption"])
        for cap in translated_captions:
            writer.writerow([cap])
        st.download_button("Download Captions as CSV", csv_fp.getvalue(), file_name="captions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error processing image: {e}") 