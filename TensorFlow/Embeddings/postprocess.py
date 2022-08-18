import tensorflow as tf
from collections import OrderedDict
import operator
from itertools import islice
import io
import csv

# Author: Christopher F. S. Maligec 250 870 443

#########################################################################
# Obtain the n-closest words in similarity to a given word.             #
# Cluster words according to their level of similarity to a given word. #
#########################################################################

class Postprocess:

    def __init__(self, vocab, v):
        self.vocab = vocab
        self.v = v
    
    def get_word_index(self, word):
        # Get row index of term-feature matrix corresponding to desired word.
        len_vocab = len(self.vocab)
        for i in range(len_vocab):
            if(self.vocab[i] == word):
        
                return i

    def get_word_tfidf(self, word):
        # From the term-feature matrix, get an array of tf-idf values corresponding to a given word.
        word_tfidf = []
        index = self.get_word_index(word)
        num_col = self.v.shape[1]
        for i in range(num_col):
            word_tfidf.append(self.v[index, i])

        return word_tfidf
        
    def get_word_vector(self, word):
        # Alternative method to above.
        return self.v[self.get_word_index(word), :]

    def get_text_vector(self, text):
        # Return the word embedding of a whole text, consisting of the sum of the vectors corresponding to all its words.
        text_vector = [0.0]*self.v.shape[0]
        for word in text:
            if(word in self.vocab):
                text_vector = tf.add(text_vector, self.self.get_word_tfidf(word)).numpy()
        return text_vector


    #######################
    ### WORD EMBEDDINGS ###
    #######################

    def generate_projector_data(self, output_dir):
        out_v = io.open(output_dir + 'vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open(output_dir + 'metadata.tsv', 'w', encoding='utf-8')
        len_vocab = len(self.vocab)
        for i in range(len_vocab):
            vec = self.get_word_tfidf(self.vocab[i])
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        for word in self.vocab:
          out_m.write(word + "\n")
        out_v.close()
        out_m.close()


    ####################
    ### DOT PRODUCTS ###
    ####################

    # The standard dot product is just xTz, where xT is x transpose, to allow for matrix multiplication of two column vectors. In TensorFlow, this is done by tf.tensordot(x, z, axes = 1).
    # If we take two column vectors with d features each, there are two corresponding vectors that contain these same features and all of their possible combinations for a total of 2^d polynomial features each. The dot product of these two latter vectors is expensive to calculate given the number of components involved, but there is a polynomial kernel equivalent that is quick to compute: (xTz + 1)^d.

    def dot(self, word, word2):
        x = self.get_word_tfidf(word)
        z = self.get_word_tfidf(word2)
        return tf.tensordot(x,z, axes=1)

    def dot_kernel(self, word, word2, kernel_degree):
        # The nth-polynomial kernel dot product.
        x = self.get_word_tfidf(word)
        z = self.get_word_tfidf(word2)
        return (tf.tensordot(x,z, axes=1) + 1)**kernel_degree
    '''
    def dot_kernel(word, word2, kernel_mode, kernel_degree):
        # The nth-polynomial kernel dot product.
        x = self.get_word_tfidf(word)
        z = self.get_word_tfidf(word2)
        return (tf.tensordot(x,z, axes=1) + kernel_mode)**kernel_degree
    '''

    def test_dots(self):
        print(f'Standard dot product of \"actor\" and \role\": {self.dot("actor", "role")}')
        for i in range(2,4):
            print(f'The {i}-polynomial kernel dot product of \"actor\" and \"role\": {self.dot_kernel("actor", "role", i)}\n')


    #########################
    ### COSINE SIMILARITY ###
    #########################

    # Calculate the similarity of a word in the corpus with every word in the corpus, including itself.
    # There are two types of similarity calculations: standard cosine similarity (sim_cos) and cosine similarity using the nth-polynomial kernel dot product (sim_cos_kernel).

    def sim_cos(self, word, word2):
        dot1 = self.dot(word, word2)
        norm_wd = tf.sqrt(self.dot(word, word))
        norm_wd2 = tf.sqrt(self.dot(word2, word2))
        return dot1/(norm_wd*norm_wd2)

    def sim_cos_kernel(self, word, word2, kernel_degree):
        dot1 = self.dot_kernel(word, word2, kernel_degree)
        norm_wd = tf.sqrt(self.dot_kernel(word, word, kernel_degree))
        norm_wd2 = tf.sqrt(self.dot_kernel(word2, word2, kernel_degree))
        return dot1/(norm_wd*norm_wd2)

    def test_similarity(self):
        # Example calculations and accuracy test: As the angle between a word embedding vector and itself should be zero, the cosine should be 1.0.
        print(f'Standard cosine similarity between \"actor\" and \"actor\": {self.sim_cos("actor", "actor")}')
        for i in range(2,4):
            print(f'The {i}-polynomial kernel cosine similarity between \"actor\" and \"actor\": {self.sim_cos_kernel("actor", "actor", i)} \n')

    def test_cosines(self):
        print(f'Standard cosine similarity between \"actor\" and \"role\": {self.sim_cos("actor", "role")}')
        for i in range(2,4):
            print(f'The {i}-polynomial kernel cosine similarity between \"actor\" and \"role\": {self.sim_cos_kernel("actor", "role", i)} \n')


    #####################
    ### GET N-CLOSEST ###
    #####################

    def take(self, n, iterable):
        # Return first n items of the iterable as a list.
        return list(islice(iterable, n))

    def get_sim_dict_standard(self, word):
        # Return a dictionary of corpus vocabulary and corresponding standard cosine similarity values for a given word parameter.
        sim_dict = {}
        if(word in self.vocab):
            for voc in self.vocab:
                sim_dict[voc] = self.sim_cos(word, voc).numpy()
        else:
            for voc in self.vocab:
                sim_dict[voc] = 0
        return sim_dict

    def get_sim_dict_kernel(self, word, kernel_degree):
        # Return a dictionary of corpus vocabulary and corresponding similarity values to a given word parameter, calculated according to a kernel-based cosine (with kernel_degree set to the desired polynomial degree).
        sim_dict = {}
        if(word in self.vocab):
            for voc in self.vocab:
                sim_dict[voc] = self.sim_cos_kernel(word, voc, kernel_degree).numpy()
        else:
            for voc in self.vocab:
                sim_dict[voc] = 0
        return sim_dict

    def n_closest_standard(self, word, n):
        # Return the top n closest vocabulary items to a word (included) along with corresponding standard cosine similarity values.
        sim_dict = self.get_sim_dict_standard(word)
        sorted_dict = dict( sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
        
        return self.take(n, sorted_dict.items())

    def n_closest_kernel(self, word, n, kernel_degree):
        # Return the top n closest vocabulary items to a word (included) along with corresponding similarity values, calculated according to a kernel-based cosine (with kernel_degree set to the desired polynomial degree).
        sim_dict = self.get_sim_dict_kernel(word, kernel_degree)
        sorted_dict = dict( sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
      
        return self.take(n, sorted_dict.items())

    def test_closest(self):
        # Example calculations
        print(f'10 closest vocabulary items for \"actor\" using normal cosine similarity: {self.n_closest_standard("actor", 10)}\n')
        for i in range(2,4):
            print(f'10 closest vocabulary items for \"actor\" using {i}-polynomial kernel cosine similarity: {self.n_closest_kernel("actor", 10, i)}\n')


    ###########################
    ### CLUSTERING OF TERMS ###
    ###########################

    def get_sim_standard(self, word):
        # For a given word parameter, return an array of standard cosine similarity values (corresponding in order to the vocabulary in vocab).
        len_vocab = len(self.vocab)
        sim_values = []
        if(word in self.vocab):
            for i in range(50): # Chosen to due to long processing time.
                sim_values.append(self.sim_cos(word, self.vocab[i]))
        else:
            sim_values = [0]*len_vocab
        return sim_values

    def get_sim_kernel(self, word, kernel_degree):
        # For a given word parameter, return an array of similarity values  (corresponding in order to the vocabulary in vocab), calculated according to a kernel-based cosine (with kernel_degree set to the desired polynomial degree).
        len_vocab = len(self.vocab)
        sim_values = []
        if(word in self.vocab):
            for i in range(50): # Chosen to due to long processing time.
                sim_values.append(self.sim_cos_kernel(word, self.vocab[i], kernel_degree))
        else:
            sim_values = [0]*len_vocab
        return sim_values

    def get_clustering_data_standard(self, word, level):
        # Cluster the vocabulary: Positives have standard cosine similarity to a given word at or above threshold level.
        data = []
        len_vocab = 50 #len(self.vocab)
        for i in range(len_vocab): # Chosen to due to long processing time.
            if(self.get_sim_standard(word)[i] >= level):
                data.append([self.vocab[i], 1])
            else:
                data.append([self.vocab[i], 0])
        return data

    def get_clustering_data_kernel(self, word, level, kernel_degree):
        # Cluster the vocabulary: Positives have similarity to a given word at or above threshold level. Similarity values are calculated according to a kernel-based cosine (with kernel_degree set to the desired polynomial degree).
        data = []
        len_vocab = 50 #len(self.vocab) 
        for i in range(len_vocab): # Chosen to due to long processing time.
            if(self.get_sim_kernel(word, kernel_degree)[i] >= level):
                data.append([self.vocab[i], 1])
            else:
                data.append([self.vocab[i], 0])
        return data

    def test_clustering(self, output_dir):
        # Datafiles for standard to cubic-polynomial-kernel similarity clustering
        standard_data = self.get_clustering_data_standard("actor", 0.5)
        quad_data = self.get_clustering_data_kernel("actor", 0.5, 2)
        cubic_data = self.get_clustering_data_kernel("actor", 0.5, 3)

        header = ["word", "label"]
        for i in range(1,4):
            with open(f'{output_dir}\\{i}_clustered_data.csv', 'w', encoding='UTF8', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                if(i == 1):
                    writer.writerows(standard_data)
                    print('Standard clustered datafile ready as 1_clustered_data.')
                elif(i == 2):
                    writer.writerows(quad_data)
                    print('Quadratic clustered datafile ready as 2_clustered_data.')
                else:
                    writer.writerows(cubic_data)
                    print('Cubic clustered datafile ready as 3_clustered_data.')

    def test_all(self, clustering_dir):
        self.test_dots()
        self.test_cosines()
        self.test_similarity()
        self.test_closest()
        self.test_clustering(clustering_dir)
