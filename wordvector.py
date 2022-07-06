import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

reviews = ['كبار السن',
        'متقدمي السن',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement']

sentiment = np.array([1,1,1,1,1,0,0,0,0,0])
print(one_hot("amazing restaurant",30))
vocab_size = 50
encoded_reviews = [one_hot(d, vocab_size) for d in reviews]
print(encoded_reviews)
max_length = 4
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')
#print(padded_reviews)
embeded_vector_size = 5
model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_length,name="embedding"))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
X = padded_reviews
y = sentiment
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print(model.summary())
#print(model.fit(X, y, epochs=50, verbose=0))
# evaluate the model
loss, accuracy = model.evaluate(X, y)
#print(accuracy)
weights = model.get_layer('embedding').get_weights()[0]
#print(len(weights))
print(weights[encoded_reviews[0]])
print(weights[encoded_reviews[1]])
#for i in range(5):
temp = np.array([((weights[encoded_reviews[0][0]] + weights[encoded_reviews[0][1]])/2),
                 ((weights[encoded_reviews[1][0]] + weights[encoded_reviews[1][1]])/2)])
print(type(((weights[encoded_reviews[0][0]] + weights[encoded_reviews[0][1]])/2)))
print(len(temp))
print(temp)