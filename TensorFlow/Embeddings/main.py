import preprocess
import embeddings_algorithms as embeddings
import postprocess
import graph_analysis as graph
import tensorflow as tf
import time
import datetime
import csv
from tqdm import tqdm
import math

def main ():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    main_path = r'C:\Users\CFSM\Desktop\Embeddings'
    data_path = main_path + r'\data'
    training_path = data_path + r'\aclImdb\train'
    sk_projector_path = main_path + r'\projector\sk_'
    sk_logs_path = main_path + r'\logs\sk'
    st_projector_path = main_path + r'\projector\st_'
    st_logs_path = main_path + r'\logs\st'
    dp_projector_path = main_path + r'\projector\dp_'
    dp_logs_path = main_path + r'\logs\dp'
    tests_path = main_path + r'\tests\sk'
    
    # PREPROCESS
    if(preprocess.download_aclImdb(url, data_path)):
        preprocess.clean_aclImdb_download(data_path)
        preprocess.filter_data(data_path + r'\aclImdb')
    
    # ALGORITHM TESTS 
    tf_train, tf_text, tf_val = preprocess.get_tf_data(training_path)
    sk_text = preprocess.get_sk_data(training_path)
    
    trunc_size = 50
    num_trials = 3
    
    max_vocab_length = 20000
    st_max_vocab = 3000
    increment = 500
    max_vocab_size = (math.floor(max_vocab_length / increment) + 1) * increment
    vocab_sizes = []
    for i in range(increment, max_vocab_size, increment):
        vocab_sizes.append(i)
    
    test_start_time = time.time()
    test_start_date = datetime.datetime.now().strftime("%Y_%m_%d__%I_%M_%S_%p")
    
    datafile = main_path + r'\timings\timing_data_' + test_start_date  + '.csv'
    
    with open(datafile, 'w', encoding='UTF8', newline = '') as f:
        row =['vocab', 'trial', 'sk', 'st', 'dp']
        writer = csv.writer(f)
        writer.writerow(row)
        for vocab_size in vocab_sizes:
            row[0] = str(vocab_size)
            for i in range(num_trials):
                row[1] = str(i + 1)
                # NOTE: sk breaks down in a for-loop writing out to a csv file at vocabulary size 3000, truncation size 29. Beyond these values, the part for sk should be commented out. However, there are no limitations when used in a for-loop printing out to console. In contrast, there do not seem to be such restrictions for st and dp. Especially st as it calculates the whole Thin SVD and then truncates it.
                print(f'sk-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                sk_vocab, sk_v = embeddings.run_sk(sk_text, vocab_size, trunc_size)
                row[2] = time.time() - old_time
                print('sk-embeddings finished \n')
                if(vocab_size <= st_max_vocab):
                    print(f'st-embeddings started: {i + 1} out of {num_trials}')
                    print(f'The number of vocabulary items in the sample: {vocab_size}')
                    old_time = time.time()
                    st_vocab, st_v = embeddings.run_st(tf_text,vocab_size, trunc_size)
                    row[3] = time.time() - old_time
                    print('st-embeddings finished \n')
                else:
                    row[3] = ''
                print(f'dp-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                dp_vocab, dp_v = embeddings.run_dp(tf_train, tf_text, tf_val, dp_logs_path, vocab_size, trunc_size)
                row[4] = time.time() - old_time
                print('dp-embeddings finished \n')
                
                writer.writerow(row)

    
    print(time.time() - test_start_time)            
                
    # POSTPROCESS
    
    vocab_size = 20000
    trunc_size = 50
    
    dp_vocab, dp_v = embeddings.run_dp(tf_train, tf_text, tf_val, dp_logs_path, vocab_size, trunc_size)   
    dp_post = postprocess.Postprocess(dp_vocab, dp_v)
    dp_post.test_all(tests_path)
    dp_post.generate_projector_data(dp_projector_path)
    
    sk_vocab, sk_v = embeddings.run_sk(sk_text, vocab_size, trunc_size)
    sk_post = postprocess.Postprocess(sk_vocab, sk_v)
    sk_post.test_all(tests_path)
    sk_post.generate_projector_data(sk_projector_path)
    
    st_vocab, st_v = embeddings.run_st(tf_text, vocab_size, trunc_size)
    st_post = postprocess.Postprocess(st_vocab, st_v)
    st_post.test_all(tests_path)
    st_post.generate_projector_data(st_projector_path)
    
    # GRAPHS
    
    datafile = main_path + r'\timings\timing_data_FINAL.csv'
    max_poly_degree = 4
    
    dp_gr = graph.GraphAnalysis(datafile) 
    dp_gr.graph_results_poly_dp('dp', max_poly_degree)
    dp_gr.graph_results_log_dp('dp')
    dp_gr.graph_results_exp_dp('dp')
    
    sk_gr = graph.GraphAnalysis(datafile) 
    sk_gr.graph_results_poly_sk('sk', max_poly_degree)
    sk_gr.graph_results_log_sk('sk')
    sk_gr.graph_results_exp_sk('sk')
    
    st_gr = graph.GraphAnalysis(datafile) 
    st_gr.graph_results_poly_st('st', max_poly_degree)
    st_gr.graph_results_log_st('st')
    st_gr.graph_results_exp_st('st')

  
if __name__ == '__main__':
    main()
