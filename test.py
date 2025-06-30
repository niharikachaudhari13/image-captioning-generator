import numpy as np
from PIL import Image
import argparse
from keras.applications.xception import Xception
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
from keras.utils import plot_model
from pickle import load
import matplotlib.pyplot as plt
import random

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
ap.add_argument('-m', '--method', default='greedy', choices=['greedy', 'beam', 'nucleus'], help="Captioning method")
args = vars(ap.parse_args())
img_path = args['image']
method = args['method']

# Constants
max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1

# Model definition
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

model = define_model(vocab_size, max_length)
model.load_weights('model-13.h5')  # or adjust to models/model_9.h5
xception_model = Xception(include_top=False, pooling='avg')

# Feature extractor
def extract_features(filename, model):
    image = Image.open(filename)
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

# Word lookup
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Caption methods
def generate_desc_greedy(model, tokenizer, photo, max_length):
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
    caption = []
    for idx in sequences[0][0]:
        word = word_for_id(idx, tokenizer)
        if word is None or word == 'end':
            break
        if word != 'start':
            caption.append(word)
    return ' '.join(caption)

def nucleus_sampling_predictions(model, tokenizer, photo, max_length, p=0.9):
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
    return ' '.join(in_text[1:])

# Run captioning
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
plt.imshow(img)

if method == 'greedy':
    caption = generate_desc_greedy(model, tokenizer, photo, max_length)
elif method == 'beam':
    caption = beam_search_predictions(model, tokenizer, photo, max_length)
elif method == 'nucleus':
    caption = nucleus_sampling_predictions(model, tokenizer, photo, max_length)
else:
    caption = "Invalid method"

print(f"\nGenerated Caption ({method}):\n{caption}")
plt.show()
