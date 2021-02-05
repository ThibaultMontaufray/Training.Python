import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,GRU,Dense

from tensorflow.keras.models import load_model

# getting data
path_to_file = "data/shakespeare.txt"
text = open(path_to_file,'r').read()
vocab = sorted(set(text))
vocab_size = len(vocab)

# text processing
char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char = np.array(vocab)

encoded_text = np.array([char_to_ind[c] for c in text])
seq_len = 120  # for that, check the max Length of one line in the file
total_num_seq = len(text) // (seq_len + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(seq_len+1,drop_remainder=True)

def create_seq_targets(seq):
    intput_txt = seq[:-1] # Hello my nam
    target_txt = seq[1:]  # ello my name
    return intput_txt,target_txt

dataset = sequences.map(create_seq_targets)

# create the batch
batch_size = 128
buffer_size = 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)

# create the model
embed_dim = 64 # we can play with to have a good result. 64 ±= vocab_size
rnn_neurons = 1026

def sparce_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true,y_pred,from_logits=True)

def create_model(vocab_size,embed_dim,rnn_neurons,batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size,embed_dim,batch_input_shape=[batch_size,None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    model.compile('adam',loss=sparce_cat_loss)
    return model

model = create_model(vocab_size,embed_dim,rnn_neurons,batch_size)
print(model.summary())

# /!\ better run the following on a gpu
#for input_example_batch, target_example_batch in dataset.take(1):
#    example_batch_predictions = model(input_example_batch)

#epochs = 30
#model.fit(dataset,epochs=epochs)

# using the model

model = create_model(vocab_size,embed_dim,rnn_neurons,batch_size=1)
model.load_weights('data/shakespeare_gen.h5')
model.build(tf.TensorShape([1,None]))

def generate_text(model,start_seed,gen_size=500,temp=1.0):
    num_generate = gen_size
    input_eval = [char_to_ind[s] for s in start_seed]
    input_eval = tf.expand_dims(input_eval,0)

    text_generated = []
    temperature = temp

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(ind_to_char[predicted_id])

    return (start_seed+"".join(text_generated))

print(generate_text(model,"JULIET",gen_size=1000))
