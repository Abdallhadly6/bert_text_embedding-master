import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text

with open('sentences.txt') as f:
    content = f.readlines()
sen = [x.strip() for x in content]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
dim = 50
f = open("vectors2.txt","w")
for j in range(len(sen)):
    for i in range(dim):
        f.write(str(tf.keras.backend.get_value(embed(sen))[j][i])+",")
    f.write("\n")
f.close()
