import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime
from tqdm.auto import tqdm
from transformers import BertTokenizer, TFBertModel

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

def pre_process_input_data(filepath='./data/concept_net/tweets.csv',encoding='cp1252',num_samples=None,random_state=None): #Change encoding if not on windows
    tweets = pd.read_csv(filepath,encoding=encoding,header=None)
    tweets.columns = ['target','id','date','flag','username','text'] #Change column names to things that make sense
    tweets = tweets.drop(columns=['id','date','flag','username']) #Remove unneeded columns from memory

    tweets = tweets.replace({'target':{0:0,4:1}}) #Dataset has only 0=negative sent, 4=positive sent, remappping to 0,1 respectivly
    if num_samples:
        tweets = tweets.groupby('target').sample(num_samples,random_state=random_state)

    return tweets

def convert_tweets_to_bert_embedding(text_arr:np.ndarray,batch_size=500):
        batches = [(i,min(i+batch_size,len(text_arr))) for i in range(0,len(text_arr),batch_size)] #Split into smaller chunks

        # _df = pd.DataFrame()
        max_twt_len = np.max([len(v) for v in text_arr])

        res = []
        print(f'Grabbing BERT Embeddings with padding to {max_twt_len} characters')
        for lower,upper in tqdm(batches):
            chunk = text_arr[lower:upper]
            features = bert_tokenizer(chunk.tolist(),padding='max_length', truncation=True, return_tensors='tf',max_length=max_twt_len)
            features = bert_model(**features).last_hidden_state[:,0,:]
            # chunk['features'] = features.numpy().tolist()
            res.append(features.numpy())
        return np.array(res).reshape(-1,768) # Reshape to output numrows by embedding space

def generate_transfer_learn_classifier():
    inputs = tf.keras.layers.Input(shape=(None,768))

    x = tf.keras.layers.Dense(128,activation = 'relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32,activation='relu')(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[inputs],outputs=x)
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

    return model

def plot_history(hist,metric,show_val=True):
    plt.plot(hist[f'{metric}'])
    if show_val:
        plt.plot(hist[f'val_{metric}'])
        plt.legend([f'Training {metric}',f'Validation {metric}'])  
    plt.show()

def generate_fine_tune_model():
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    token_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    for l in bert_model.bert.encoder.layer:
        l.trainable = False
    bert_model.bert.embeddings.trainable = False
    # bert_model.bert.trainable = False
    bert_model.bert.encoder.layer[-1].trainable = True
    # BERT model
    bert_model.bert.pooler.trainable = False
    outputs = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # Extract the pooled output (CLS token) from BERT
    pooled_output = outputs.pooler_output

    # Add your custom top layer
    dense_layer = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
    dropout_layer = tf.keras.layers.Dropout(0.5)(dense_layer)
    dropout_layer = tf.keras.layers.Dense(32,activation='relu')(dropout_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dropout_layer)

    # Define model
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output_layer)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.losses.binary_crossentropy,
                metrics=['accuracy'])
    return model
def pre_process_bert_inputs(x:np.ndarray,y:np.ndarray):
    sentences = x
    sentiment_values = y  # Example sentiment values: 1 for positive, 0 for negative

    # Tokenize input sentences
    tokenized_inputs = bert_tokenizer(sentences.tolist(), padding=True, truncation=True, return_tensors="tf")

    # Convert sentiment values to TensorFlow tensors
    sentiment_values = tf.convert_to_tensor(sentiment_values)

    # Prepare input data as a dictionary
    input_data = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "token_type_ids": tokenized_inputs["token_type_ids"]
    }
    return input_data,sentiment_values

if __name__ == "__main__":
    mdl = generate_fine_tune_model()
    print(mdl.summary())