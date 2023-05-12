import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Dense

# importing and reading the data
with open('superhero/superheroes.txt', 'r') as f:
    data = f.read()

# create a tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
    split='\n',
)
tokenizer.fit_on_texts(data)

# creating dictionaries
char_to_number = tokenizer.word_index
number_to_char = dict((v, k) for k, v in char_to_number.items())

# creating a list of names that doesn't contain the '\n' character
names = data.splitlines()
print(names[:10])


# converting names to sequences
def name_to_sequence(name):
    return [tokenizer.texts_to_sequences(c)[0][0] for c in name]


print(name_to_sequence(names[0]))


# converting sequences to names
def sequence_to_name(seq):
    return ''.join(number_to_char[i] for i in seq)


sequence_to_name(name_to_sequence(names[0]))

# creating sequences
sequences = []
for name in names:
    seq = name_to_sequence(name)
    if len(seq) >= 2:
        sequences += [seq[:i] for i in range(2, len(seq) + 1)]

# getting the maximum length of names in the dataset
max_len = max([len(seq) for seq in sequences])

# padding all sequences to be of a length of max_len
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, padding='pre',
    maxlen=max_len
)
print(padded_sequences[0])
print(padded_sequences.shape)

#  splitting padded sequences into examples and labels
x, y = padded_sequences[:, :-1], padded_sequences[:, -1]
print(x.shape, y.shape)

# splitting examples into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# number of chars in our vocabulary
max_number_of_chars = len(char_to_number) + 1
print(max_number_of_chars)

# creating the model
model = Sequential([
    Embedding(max_number_of_chars, 8, input_length=max_len - 1),
    Conv1D(64, 5, strides=1, activation='tanh', padding='causal'),
    MaxPool1D(2),
    LSTM(32),
    Dense(max_number_of_chars, activation='softmax')
])

# compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# training the model
fit = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor=('val_accuracy'), patience=3)]
)


# generating names
def generate_names(seed):
    for i in range(40):
        seq = name_to_sequence(seed)
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences([seq], padding='pre', maxlen=max_len - 1, truncating='pre')
        prediction = model.predict(padded_seq)[0]
        pred_char = number_to_char[tf.argmax(prediction).numpy()]
        seed += pred_char
        if pred_char == '\t':
            break
        print(seed)


generate_names('a')




