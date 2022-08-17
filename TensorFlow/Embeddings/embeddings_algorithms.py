import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tensorflow import linalg
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, InputLayer
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import tensorflow_datasets as tfds


class SVDLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(SVDLayer, self).__init__()

        def call(self, inputs):
            _, _, v = tf.linalg.svd(inputs)
            return v
          
def run_sk(text_data, vocab_size, trunc_size):

    # Turn the corpus text into a document-term sparse matrix of tf-idf values.
    vectorizer = TfidfVectorizer(max_features=vocab_size, lowercase=False)
    vectorized_corpus = vectorizer.fit_transform(text_data)
    
    # Obtain the corpus vocabulary.
    vocab = vectorizer.get_feature_names_out()
    
    # Apply Truncated Singular Value Decomposition (Truncated SVD) to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix M is approximately decomposed into U_t*Sigma_t*V_transpose_t.
    # If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U_t is mxt, Sigma_t is txt, and V_transpose_t is txn. The latent dimension t = the number of features after truncation (trunc_size).

    # Reduce the dimensionality of the document-term sparse matrix.
    lsa_obj = TruncatedSVD(n_components=trunc_size, random_state=123)
    lsa_obj.fit_transform(vectorized_corpus)

    # Obtain feature-term matrix of shape trunc_size x len_vocab. This can then be used to define word embeddings with a much lower number of dimensions.
    v = lsa_obj.components_

    #Obtain term-feature matrix of shape len_vocab x trunc_size.
    v = np.transpose(v)

    return vocab, v

def run_st(text_data, vocab_size, trunc_size):

    vectorize_layer=TextVectorization(
        standardize=None,
        max_tokens=vocab_size,
        output_mode='tf-idf'
        )

    # Transform text data to adapt() format.
    vectorize_layer.adapt(text_data)
    vocab = vectorize_layer.get_vocabulary(include_special_tokens = False)
    vectorized_corpus = vectorize_layer(text_data)
    
    # Apply Thin Singular Value Decomposition (Thin SVD) to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix (M) is decomposed into U*Sigma*V_transpose.
    # If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U is mxk, Sigma is kxk, and v_transpose is kxn where k = min{m, n}. We can call k the number of features.
    # The term-feature matrix, which is just v with dimensions nxk, can then be used to define word embeddings with a much lower number of dimensions.
    _, _, v = tf.linalg.svd(vectorized_corpus)
    v = v.numpy()
    
    # Truncate v to lower dimensions, so that v has dimensions nxt with t << k.
    v = v[:, 0:trunc_size]

    return vocab, v
    
def run_dp(train_data, text_data, val_data, logs_path, vocab_size, trunc_size, use_profiler=False):

    # Use the text vectorization layer to map strings to integers.
    # Set sequence length as all samples are not of the same length.
    sequence_length=100
    vectorize_layer=TextVectorization(
        standardize=None,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
        )

    # Get the vocabulary.
    vectorize_layer.adapt(text_data)
    vocab = vectorize_layer.get_vocabulary(include_special_tokens = False)

    # Create model.
    embedding_dim=trunc_size
    model=Sequential([
      vectorize_layer,
      Embedding(vocab_size, embedding_dim, name="embedding"),
      GlobalAveragePooling1D(),
      Dense(embedding_dim, activation='relu'),
      Dense(1)
    ])  
    
    # Compile model.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model.
    if(use_profiler):
    
        # To make use of the TensorBoard Profiler to analyze results for further optimization. Callbacks can only be included in model fits, which is another reason for having created a model to vectorize the text data.
        tensorboard_callback=tf.keras.callbacks.TensorBoard(
            log_dir=logs_path, 
            profile_batch=(1,50)
            )
        
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            #verbose=0, In case one does not want progress bars.
            callbacks=[tensorboard_callback]
            )
    else:
        model.fit(
            train_data,
            epochs=10
            #verbose=0 In case one does not want progress bars.
            )

    # Get the embeddings.
    v = model.get_layer('embedding').get_weights()[0]

    return vocab, v

def test1():
    
    import preprocess
    import emb as embeddings
    import postprocess
    import time
    import datetime
    import csv
    from tqdm import tqdm
    import math
    import graph_analysis as graph
    import tensorflow as tf
    
    main_path = r'C:\Users\CFSM\Desktop\Embeddings'
    data_path = main_path + r'\data'
    training_path = data_path + r'\aclImdb\train'
    dp_logs_path = main_path + r'\logs\dp'
    
    tf_train, tf_text, tf_val = preprocess.get_tf_data(training_path)
    sk_text = preprocess.get_sk_data(training_path)
    
    vocab_size = 100#20000
    trunc_size = 10#50
    num_trials = 2#3
    
    max_vocab_length = 100#20000
    st_max_vocab = 50#5000
    increment = 20#500
    max_vocab_size = (math.floor(max_vocab_length / increment) + 1) * increment
    vocab_sizes = []
    
    test_start_time = time.time()
    test_start_date = datetime.datetime.now().strftime("%Y_%m_%d__%I_%M_%S_%p")
    
    logname = main_path + r'\timings\timing_log_' + test_start_date  + '.csv'
    for i in range(increment, max_vocab_size, increment):
        vocab_sizes.append(i)
    with open(logname, 'w', encoding='UTF8', newline = '') as f:
        row =['vocab', 'trial', 'sk', 'st', 'dp']
        writer = csv.writer(f)
        writer.writerow(row)
        for vocab_size in vocab_sizes:
            row[0] = str(vocab_size)
            for i in range(num_trials):
                row[1] = str(i + 1)
                print(f'sk-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                sk_vocab, sk_v = run_sk(sk_text, vocab_size, trunc_size)
                row[2] = time.time() - old_time
                print('sk-embeddings finished \n')
                if(st_max_vocab >= vocab_size):
                    print(f'st-embeddings started: {i + 1} out of {num_trials}')
                    print(f'The number of vocabulary items in the sample: {vocab_size}')
                    old_time = time.time()
                    st_vocab, st_v = run_st(tf_text, vocab_size, trunc_size)
                    row[3] = time.time() - old_time
                    print('st-embeddings finished \n')
                else:
                    row[3] = ''
                    print('st-embeddings limit exceeded \n')
                print(f'dp-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                dp_vocab, dp_v = run_dp(tf_train, tf_text, tf_val,dp_logs_path, vocab_size, trunc_size, use_profiler=False)
                row[4] = time.time() - old_time
                print('dp-embeddings finished \n')
                writer.writerow(row)
    

    #TOTAL_VOCAB = 54621#(sk) #54653#(dp)
    #TOTAL_DOCS = 20000
 
if __name__ == '__main__':
    test1()    
 