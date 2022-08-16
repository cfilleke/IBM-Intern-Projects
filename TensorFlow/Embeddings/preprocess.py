import tensorflow as tf
from tensorflow import keras
import os
from os import listdir
from os.path import isfile, isdir, join
from tqdm import tqdm
from urllib.parse import urlparse
import shutil
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def download_aclImdb(url, data_path):
    acl_path = data_path + r'\aclImdb'
    is_download = False
    if(not os.path.exists(acl_path)):
        a = urlparse(url)
        filename = os.path.basename(a.path)
        dataset = tf.keras.utils.get_file(filename, url,
                                          untar=True, cache_dir=data_path,
                                          cache_subdir='')
        is_download = True
        
    return is_download

def clean_aclImdb_download(data_path):
    print("Removing unnecessary files")
    # Custom remove unwanted dirs and files.
    dataset_dir = data_path + r'\aclImdb'
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    remove_dir = os.path.join(train_dir, 'unsup')
    os.remove(data_path + r'\aclImdb_v1.tar.gz')
    shutil.rmtree(remove_dir)
    for f in os.listdir(dataset_dir):
        p = dataset_dir + '\\' + f
        if os.path.isfile(p): 
            os.remove(p)
    for f in os.listdir(train_dir):
        p = train_dir + '\\' + f
        if os.path.isfile(p): 
            os.remove(p)
    for f in os.listdir(test_dir):
        p = test_dir + '\\' + f
        if os.path.isfile(p): 
            os.remove(p)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.tag.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def filter_file(file_path):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # opening the text file
    #with open(file_path,'r+', encoding='cp437', errors='ignore') as file:
    with open(file_path,'r+', encoding='utf-8', errors='ignore') as file:
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(file.read())
        filtered = []   
        for word in words:
            if(not any(ch.isdigit() for ch in word)):
                word = word.lower()
                if(word not in stop_words and word != "br"):
                    pos = get_wordnet_pos(word)
                    word = lemmatizer.lemmatize(word, pos)
                    filtered.append(word)
        text = ' '.join(filtered)
        file.truncate(0)
        file.seek(0)
        file.write(text)

def filter_data(data_dir):
    print(f"Processing files in {data_dir}")
    for f in tqdm(listdir(data_dir)):
        path = join(data_dir, f)
        if isfile(path):
            filter_file(path)
        elif isdir(path):
            filter_data(path)

# Training and validation text data                                


def get_tf_data(training_path):
    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(
        training_path, batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(
        training_path, batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed) 
    train_ds = train_ds.shuffle(buffer_size = 100, seed = seed, reshuffle_each_iteration = False)
    # Make a text-only dataset (no labels).   
    text_ds = train_ds.map(lambda x, y: x)
    text_ds = text_ds.unbatch()
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #text_ds = text_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    text_ds = list(text_ds.as_numpy_iterator())

    return train_ds, text_ds, val_ds

def get_sk_data(training_path):
    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(
        training_path, validation_split=0.2,
        subset='training', batch_size=batch_size, seed=seed)
    train_ds = train_ds.unbatch()
    
    # Make a text-only dataset (no labels).
    train_ds = train_ds.map(lambda x, y: x)

    return list(train_ds.as_numpy_iterator())

if __name__ == '__main__':
    main_path = r'C:\Users\CFSM\Desktop\Embeddings'
    data_path = main_path + r'\data'
    training_path = data_path + r'\aclImdb\train'
    
    x, y, z = get_tf_data(training_path)
    #print(tf.data.experimental.cardinality(x).numpy(), tf.data.experimental.cardinality(y).numpy(), tf.data.experimental.cardinality(z).numpy())
    y = y.unbatch()
    y = list(y.as_numpy_iterator())
    print(y)
    #print(tf.data.experimental.cardinality(x).numpy())
    #x = list(x.as_numpy_iterator())
    #print(len(x))
    #print(len(get_sk_data(training_path)))