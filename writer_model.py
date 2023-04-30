import tensorflow as tf
import numpy as np
import os
import pickle
# from keras.utils import pad_sequences
# import d6tflow


# Load the dataset of your personal writing to train a model to write in your style
data_dir = "/writing"
files = os.listdir(data_dir)


def load_corpus_data():
    corpus_data = []
    for filename in files:
        if filename.endswith(".pkl"):
            with open(os.path.join("corpus", filename), "rb") as f:
                data = pickle.load(f)
                corpus_data.extend(data)
                data = np.array(data)
    return data


x_train = []
y_train = []
corpus = load_corpus_data()
for i, file in enumerate(corpus):
    with open(os.path.join(data_dir, file), 'r') as f:
        text = f.read()
        x_train.append(text)
        y_train.append(i)

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 64, input_length=100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(files), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    verbose=1,
                    validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
print('\nTest accuracy:', test_acc)

# Define a task for generating text from the trained model and saving it to a file


def generate_text_and_save(seed_text, filename):
    generated_text = seed_text
    while True:
        predictions = history.predict(x_train, verbose=0)[0]
        predicted_word_index = np.argmax(predictions)
        predicted_word = tokenizer.index_word[predicted_word_index]
        generated_text += " " + predicted_word
        if predicted_word == "." or len(generated_text) >= 1000:
            break
    with open(f"data/pkl/text/{filename}.pkl", "wb") as f:
        pickle.dump(generated_text, f)
    return f"data/text/{filename}.txt"


