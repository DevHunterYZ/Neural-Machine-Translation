from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
# Using TensorFlow backend.

lines = open('fra.txt', encoding='utf-8').read().split('\n')
eng_sent = []
fra_sent = []
eng_chars = set()
fra_chars = set()
nb_samples = 10000

# İngilizce ve Fransızca cümleleri işle.
for line in range(nb_samples):
    
    eng_line = str(lines[line]).split('\t')[0]
    
    # Cümlenin başlangıcı için '\ t' ve cümlenin sonunu belirtmek için '\ n' ekleyin
    fra_line = '\t' + str(lines[line]).split('\t')[1] + '\n'
    eng_sent.append(eng_line)
    fra_sent.append(fra_line)
    
    for ch in eng_line:
        if (ch not in eng_chars):
            eng_chars.add(ch)
            
    for ch in fra_line:
        if (ch not in fra_chars):
            fra_chars.add(ch)

fra_chars = sorted(list(fra_chars))
eng_chars = sorted(list(eng_chars))

# Her ingilizce karakteri indekslemek için sözlük - anahtar indeksi ve değeri ingilizce karakter
eng_index_to_char_dict = {}

# İndeksi verilen ingilizce karakter almak için sözlük - anahtar ingilizce karakter ve değer endeksi
eng_char_to_index_dict = {}

for k, v in enumerate(eng_chars):
    eng_index_to_char_dict[k] = v
    eng_char_to_index_dict[v] = k
    
# Her fransız karakterini endekslemek için sözlük - anahtar dizin ve değer fransız karakter
fra_index_to_char_dict = {}

# Dizini verilen Fransız karakter almak için sözlük - anahtar Fransız karakter ve değer dizin
fra_char_to_index_dict = {}
for k, v in enumerate(fra_chars):
    fra_index_to_char_dict[k] = v
    fra_char_to_index_dict[v] = k

max_len_eng_sent = max([len(line) for line in eng_sent])
max_len_fra_sent = max([len(line) for line in fra_sent])


tokenized_eng_sentences = np.zeros(shape = (nb_samples,max_len_eng_sent,len(eng_chars)), dtype='float32')
tokenized_fra_sentences = np.zeros(shape = (nb_samples,max_len_fra_sent,len(fra_chars)), dtype='float32')
target_data = np.zeros((nb_samples, max_len_fra_sent, len(fra_chars)),dtype='float32')

# İngilizce ve fransızca cümleleri vektör haline getirelim.

for i in range(nb_samples):
    for k,ch in enumerate(eng_sent[i]):
        tokenized_eng_sentences[i,k,eng_char_to_index_dict[ch]] = 1
        
    for k,ch in enumerate(fra_sent[i]):
        tokenized_fra_sentences[i,k,fra_char_to_index_dict[ch]] = 1

        # decoder_target_data will be ahead by one timestep and will not include the start character.
        if k > 0:
            target_data[i,k-1,fra_char_to_index_dict[ch]] = 1

# Encoder modeli

encoder_input = Input(shape=(None,len(eng_chars)))
encoder_LSTM = LSTM(256,return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
encoder_states = [encoder_h, encoder_c]

# Decoder modeli

decoder_input = Input(shape=(None,len(fra_chars)))
decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(fra_chars),activation='softmax')
decoder_out = decoder_dense (decoder_out)

model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])

# Train edelim.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[tokenized_eng_sentences,tokenized_fra_sentences], 
          y=target_data,
          batch_size=64,
          epochs=50,
          validation_split=0.2)
# Test için Çıkarım(Inference) modelleri

# Encoder Çıkarım modeli
encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder Çıkarım modeli
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, 
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]

decoder_out = decoder_dense(decoder_out)

decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )    
def decode_seq(inp_seq):
    
    # İlk durumlar(states) değeri kodlayıcıdan geliyor.
    states_val = encoder_model_inf.predict(inp_seq)
    
    target_seq = np.zeros((1, 1, len(fra_chars)))
    target_seq[0, 0, fra_char_to_index_dict['\t']] = 1
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_fra_char = fra_index_to_char_dict[max_val_index]
        translated_sent += sampled_fra_char
        
        if ( (sampled_fra_char == '\n') or (len(translated_sent) > max_len_fra_sent)) :
            stop_condition = True
        
        target_seq = np.zeros((1, 1, len(fra_chars)))
        target_seq[0, 0, max_val_index] = 1
        
        states_val = [decoder_h, decoder_c]
        
    return translated_sent

for seq_index in range(10):
    inp_seq = tokenized_eng_sentences[seq_index:seq_index+1]
    translated_sent = decode_seq(inp_seq)
    print('-')
    print('Girdi(Input) cümlesi:', eng_sent[seq_index])
    print('Deşifrelenmiş(Decoded) cümle:', translated_sent)
          
print(fra_sent)
