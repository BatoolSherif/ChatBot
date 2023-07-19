import random
import json

import pickle
# Pickle can be used to save trained models or other important data structures to disk,
# so that they can be easily loaded and reused later.

import numpy as np
import tensorflow as tf

# colorama to print colored text
from colorama import Fore

import nltk
from nltk.stem import WordNetLemmatizer


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('D:\Batool\chatbot\intents.json').read())

words = []  # a list of all of the words of our patterns
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)

        # The extend() method is used to add multiple elements to the end of a list.
        words.extend(wordList)

        # The append() method is used to add a single element to the end of a list.
        documents.append((wordList, intent['tag']))

        # Checking if intent['tag'] is already in classes, and if not, appending it to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# "bag" is a list which will contain the numerical representation of the sentence.

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    # bag --> indicates whether that word exists in the document or not. 
    # The position of the element in the bag list corresponds to the index of the word in the words list.
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    #classes.index(document[1])--> find index of class from document [word , class]
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)] #bag
trainY = training[:, len(words):] #outputrow


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')



