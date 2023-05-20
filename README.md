# Superhero-name-generator-using-TensorFlow
Content:
Introduction
Run Locally
Data And Tokenizer
Converting names to sequences and vice versa
Creating Examples
Creating the model
Compiling the model
Training the model
Generating names
In this project, we're going to create a neural network model and train it on a data set of 9000 superhero and supervillain names from different comic books, TV shows and movies in order to learn to generate similar ones.

Introduction
We're going to create a character level language model that will predict the next character for a given sequence.

In order to get a superhero name, we will need to give our trained model some sort of seed input, which can be a single character from a sequence of characters, and the model will then generate the next character and add it to the seed input to create a new input, which is then used again to generate the next character and so on.

Run Locally
Clone the project

  git clone https://github.com/superrae/Superhero-name-generator-using-TensorFlow.git
Go to the project directory

  cd Superhero-name-generator-using-TensorFlow
Importing Training data:

    git clone https://github.com/am1tyadav/superhero 
Install dependencies (Optional : Create a Virtual Environment for this project)

  pip install tensorflow
  pip install scikit-learn
Run the project

  python main.py
or

  python3 main.py
Data and tokenizer
I- Reading the data:
Reading:

with open('superhero/superheroes.txt', 'r') as f:
    data = f.read()
print(data[:100])
Notes:

The first line uses the open() function to open the file in read mode ('r') and assigns the file object to the variable f.
The second line reads the entire contents of the file object f using the read() method and assigns it to the variable data.
The third line extracts the first 100 characters of data using the slicing operator and prints them to the console. with is a Python statement that is used to create a context in which a file or a resource is used. It is typically used along with the open() function to open a file and ensure that the file is closed when the block of code inside the with statement finishes executing.
II- Creating a Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(
   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
   split='\n',
)
tokenizer.fit_on_texts(data)
A tokenizer is used to give a numeric representation to all the character items in our vocabulary so that they’re understood by the model.

Notes:

The first line creates an instance of the Tokenizer class and assigns it to the variable tokenizer.
The Tokenizer class takes several arguments, which are passed as keyword arguments, in this case:
The filters argument specifies a string containing characters to be filtered out from the text.
The split argument specifies the string to be used for splitting the text into tokens. In this case, the string is set to '\n'.
III- Creating a char-to-index dictionary and an index-to-char dictionary
  character_to_index = tokenizer.word_index
  index_to_character = dict((v, k) for k, v in
                       character_to_index.items())
  print(index_to_character)
Notes:

word_index is an attribute of the tokenizer class and it is a dictionary that maps words (or tokens) to their corresponding integer indices in the vocabulary.
During training, the fit_on_texts() method of the Tokenizer class is used to build the word_index attribute by analyzing the frequency of each unique word in the text corpus. The most frequent words are assigned lower integer indices, while less frequent words are assigned higher integer indices.
char_to_number.items() returns a list of key-value pairs from the dictionary, where each key-value pair is represented as a tuple (key, value).
Converting names to sequences and vice versa
We need to convert the names stored in our dataset into sequences of numbers before they’re fed into the model.

  names = data.splitlines() 
I- converting names to sequences
  def name-to-sequence(name):
      return [tokenizer.texts_to_sequences(c)[0][0] for c in name] 
II- converting sequences to names
  def sequence_to_name(seq):
     return ''.join([index_to_char[i] for i in seq if i != 0])
Notes:

The splitlines() method is called on the data variable to split it into a list of strings, where each string represents a line of text in data.
The split='\n' argument in the Tokenizer constructor tells the tokenizer to split the input text into separate lines based on newline characters (\n). However, it doesn't remove the newline characters themselves from the lines.
Therefore, the names = data.splitlines() line of code is used to remove the newline characters from each line of the input text and create a list of cleaned-up lines.
Creating examples
I- Creating sequences:
sequences = []
for name in names:
sequences = []
for name in names:
   seq = name_to_sequence(name)
   if len(seq) >= 2:
       sequences += [seq[:i] for i in range(2, len(seq) + 1)]
Notes:

sequences have to be at least 2 characters long so that the first
character is the input and the second one is the label.
II- Getting the maximum length of all names
  max_len = max([len(name) for seq in sequences])
III- Padding sequences
We’re padding sequences because we need the inputs to be of fixed-length in order to be fed into our model.

padded_sequences = tf.keras.preprocessing.sequence.pad_sequence
(
   sequences, padding = 'pre',
   maxlen = max_len
)
Notes:

sequences: A list of sequences of integers to be padded.
all sequences in the list must have the same length.
padding: A string indicating whether the padding should be added to the beginning of the sequence (padding='pre') or the end of the sequence (padding='post').
maxlen: The maximum length of the padded sequence. If a sequence is shorter than maxlen, it will be padded with zeros at the beginning or end (depending on the value of padding) until it reaches the maximum length. If a sequence is longer than maxlen, it will be truncated so that it only contains the first maxlen elements.
IV- Splitting padded sequences to examples and labels
x, y = padded_sequences[:, :-1], padded_sequences[:, -1]
Note:

The : in padded_sequences[:, :-1] specifies that we want to extract all rows of the padded_sequences array. The :-1 specifies that we want to exclude the last column of each row. This is achieved by using the colon : to select all columns, and the -1 to specify the last column, but exclude it.
When using NumPy slicing syntax to slice a multidimensional array, you need to use a comma to separate the indexing or slicing expressions for each dimension of the array.
The difference between padded_sequences[:, :-1] and padded_sequences[:, -1] is that the former extracts all columns except the last column from all rows of the padded_sequences array, while the latter extracts only the last column from all rows of the padded_sequences array.
V- Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split 
Note: Default split is 70% for the training set and 30% for the test set.

Creating the model
model = Sequential([
   Embedding(max_number_of_chars, 8,
             input_length=max_len - 1),
   Conv1D(64, 5, strides=1, activation='tanh',
   padding='causal'),
   MaxPool1D(2),
   LSTM(32),
   Dense(max_number_of_chars, activation='softmax')
])
The model is built using the Sequential API that allows for building a linear stack of layers in a neural network. The model architecture consists of the following layers, in order:

Embedding layer: This layer maps each character in the input sequence to a fixed-size vector of dimension 8. The max_number_of_chars parameter specifies the maximum number of distinct characters in the input. The input_length parameter specifies the length of the input sequence, which is max_len - 1, where max_len is the maximum length of names in the dataset. Note: it’s max_len - 1 because that one is used to represent the label.

1D convolutional layer:

This layer contains 64 filters, each filter is of fixed size, 5. Each filter moves along the input sequence so that a dot product is computed between the filter weights and a local window of the input at each position. This operation produces a new output sequence, where each element represents the convolution of the filter with the input at a particular position.

The size of the filter determines the number of neighboring elements in the input sequence that are considered at each position.

The stride parameter determines the amount of movement of the filter along the input sequence at each step.

The activation function applied after the convolutional operation introduces non-linearity in the output and allows the model to learn complex patterns and relationships in the input data.

padding='causal' specifies the type of padding to apply to the input sequence before the convolution operation. In this case, 'causal' padding is used, which pads the sequence with zeros in such a way that the output at each position depends only on the input values up to that position and does not "leak" information from future positions.

MaxPool1D:

This layer operates by sliding a window of fixed size -determined by the pool_size parameter - over the input sequence - which is the output of the Conv1D layer - and selecting the maximum value in each window. By selecting the maximum value, the layer retains the most important feature in each window, while discarding the less important information.
Note: In this case, the 1D convolutional layer is designed to extract local patterns and features from the input sequence, while the MaxPool1D layer is designed to select the most important features and reduce the dimensionality of the output.
LSTM

This layer is designed to capture long-term dependencies in the input sequence -which is the output of the MaxPool1D layer- and produce a corresponding output sequence, which can then be used for prediction or classification.
Dense

Maps the output of the LSTM layer to a vector of dimension max_number_of_chars. The softmax activation function normalizes the output vector to represent a probability distribution over the possible characters in the output.
Compiling the model:
model.compile(
   optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)
Parameters of compile method:
Optimizer: This parameter specifies the optimization algorithm used to update the weights of the neural network during training.
Loss: This parameter specifies the loss function that is used to measure the difference between the predicted output of the neural network and the true output during training.
Note:

The main difference between sparse_categorical_crossentropy and categorical_crossentropy is in the format of the true class labels. categorical_crossentropy is used when the true class labels are one-hot encoded vectors, where each vector has a length equal to the number of classes, and the index corresponding to the true class label is set to 1, while all other indices are set to 0. For example, in a classification task with 3 classes, the true label for a sample that belongs to class 2 would be represented as [0, 1, 0].
On the other hand, sparse_categorical_crossentropy is used when the target labels are integers, where each integer represents the index of the true class. For example, in a classification task with 3 classes, the true label for a sample that belongs to class 2 would be represented as 2.
Metrics: This parameter specifies the evaluation metric used to monitor the performance of the neural network during training.
Training the model
   x_train, y_train,
   validation_data=(x_test, y_test), epochs=50, verbose=2, 
   callbacks = [tf.keras.callbacks.EarlyStopping(
                                   monitor='val_accuracy'), patience=3)]
)
x_train and y_train specify the training dataset, where x_train is the input data and y_train is the corresponding target labels. The fit method uses this data to train the neural network and adjust the weights of the network to minimize the loss function.
validation_data specifies a separate validation dataset to evaluate the performance of the neural network during training. The validation dataset consists of x_test, which is the input data, and y_test, which is the corresponding target labels.
Epoch is an iteration during which the neural network processes the entire training dataset, computes the loss function, and updates the weights of the network using backpropagation.
Verbose parameter is used to control the amount of logging output during the training process.
Callbacks parameter specifies a list of callbacks to be applied during training. In this code snippet, the EarlyStopping callback is used to stop training early if the validation accuracy does not improve for 3 consecutive epochs.
Generating names
def generate_names(seed):
   for i in range(40): 
       seq = name_to_sequence(seed)
       padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
       [seq], padding='pre', maxlen=max_len - 1, truncating='pre')
       
       prediction = model.predict(padded_seq)[0]
       pred_char = number_to_char[tf.argmax(prediction).numpy()]
       seed += pred_char

       if pred_char == '\t':
           break
       print(seed)
Notes:

The function passes the padded_seq sequence to the trained neural network model using the model.predict method. This returns a prediction for the next character in the sequence, which is stored in the variable prediction.
The function selects the character with the highest predicted probability using the tf.argmax function and stores it in the list pred_char.
The function appends the pred_char character to the end of the seed string to create a new seed for the next iteration of the loop.
If the predicted character is the tab character (\t), the function breaks out of the loop.
After the loop has finished iterating, the function prints the final seed string.
