import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
'''
tensorflow==2.0.0
tensorflow-estimator==2.0.1
tensorflow-hub==0.7.0
'''
# load universal sentence encoder module
def load_USE_encoder(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
# load the encoder module
encoder = load_USE_encoder('./USE')
# define some messages
message1 = ["كيف حالك ياخي هل انت بخير ؟"]
message2 = ["كيف حالك ياخي هل انت بخير ؟"]
# encode the messages
print( (encoder(message1)[0])[0])
print("-----------------------------")
print( encoder(message2))
