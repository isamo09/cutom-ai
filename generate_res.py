from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = load_model("model.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def generate_text(_prompt: str):
    for _ in range(10):
        token_list = tokenizer.texts_to_sequences([_prompt])[0]
        token_list = pad_sequences([token_list], maxlen=6, padding="pre")
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        _prompt += " " + output_word
    return _prompt

prompt = input("?pLieZZ>>")
print(generate_text(prompt))
