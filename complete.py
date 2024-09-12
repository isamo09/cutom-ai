from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import wikipedia

import numpy as np
import tensorflow as tf

dataset_name = "dataset.txt"
dataset = open(file=dataset_name, mode="r").read()
wikipedia.set_lang("ru")


def get_wikipedia_texts_by_keyword(keyword, num_pages=5):
    search_results = wikipedia.search(keyword, results=num_pages)
    texts = []
    for page_name in search_results:
        try:
            content = wikipedia.page(page_name).content
            texts.append(content)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Слишком много значений для '{page_name}': {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Страница '{page_name}' не найдена.")
    return texts

keyword = "машинное обучение"
num_pages = 15
wikipedia_texts = get_wikipedia_texts_by_keyword(keyword, num_pages)
all_texts = " Ты искуственный интелект отвечающий на вопрос ".join(wikipedia_texts)
complete_data = all_texts + dataset

tokenizer = Tokenizer()
tokenizer.fit_on_texts([complete_data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in complete_data.split("."):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 64))
model.add(LSTM(100))
model.add(Dense(total_words, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X, y, epochs=7, verbose=1)

model.save("model.h5")

import pickle
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model is Complete.")
