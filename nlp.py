from tensorflow.keras.models import load_model

model = create_model(vocab_size,embed_dim,rnn_neurons,batch_size=1)
model.load_weights('data/shakespeare_gen.h5')
model.build(tf.TensorShape([1,None]))

print(model.summary())

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
