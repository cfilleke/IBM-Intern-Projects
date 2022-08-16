import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

# For information on polynomial features in linear regression, see https://data36.com/polynomial-regression-python-scikit-learn/
# Also see for documentation https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
# "Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]."

class GraphAnalysis:

    def __init__(self, datafile):
        self.datafile = datafile
    
    def graph_analysis_poly(self, alg, deg):
        df = pd.read_csv(self.datafile)
        #df.describe()
        run_time = df.loc[:, alg]
        voc = df.loc[:, 'vocab'].to_numpy()
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        voc_poly = poly.fit_transform(voc.reshape(-1, 1))
        Voc_poly = np.c_[np.ones(voc_poly.shape), voc_poly]
        model = RANSACRegressor(random_state=123)
        model.fit(Voc_poly, run_time) 
        run_time_pred = model.predict(Voc_poly)
        return voc, Voc_poly, run_time, run_time_pred, model
    
    def graph_analysis_log(self, alg):
        df = pd.read_csv(self.datafile)
        #df.describe()
        run_time = df.loc[:, alg]
        voc = df.loc[:, 'vocab'].to_numpy()
        voc_log = np.log(voc)
        poly = PolynomialFeatures(degree=1, include_bias=False)
        voc_poly_log = poly.fit_transform(voc_log.reshape(-1, 1))
        Voc_poly_log = np.c_[np.ones(voc_poly_log.shape), voc_poly_log]
        model = RANSACRegressor(random_state=123)
        model.fit(Voc_poly_log, run_time) 
        run_time_pred = model.predict(Voc_poly_log)
        return voc_log, Voc_poly_log, run_time, run_time_pred, model
        
    def graph_analysis_exp(self, alg):
        df = pd.read_csv(self.datafile)
        #df.describe()
        run_time = df.loc[:, alg]
        run_time_log = np.log(run_time)
        voc = df.loc[:, 'vocab'].to_numpy()
        poly = PolynomialFeatures(degree=1, include_bias=False)
        voc_poly = poly.fit_transform(voc.reshape(-1, 1))
        Voc_poly = np.c_[np.ones(voc_poly.shape), voc_poly]
        model = RANSACRegressor(random_state=123)
        model.fit(Voc_poly, run_time_log) 
        run_time_pred = model.predict(Voc_poly)
        return voc, Voc_poly, run_time_log, run_time_pred, model
    
    def gen_graph_poly(self, alg, deg, graph_name):   
        voc, _, run_time, run_time_pred, _ = self.graph_analysis_poly(alg, deg)
        plt.figure(figsize = (10,6))
        plt.scatter(voc, run_time)
        plt.plot(voc, run_time_pred, c = 'red')
        plt.title(f'{alg}_algorithm_poly – vocab vs. running time (sec)')
        plt.xlabel('Vocabulary') 
        plt.ylabel('Running time (sec)')
        plt.savefig(graph_name)
    
    def gen_graph_log(self, alg, graph_name):   
        voc, _, run_time, run_time_pred, _ = self.graph_analysis_log(alg)
        plt.figure(figsize = (10,6))
        plt.scatter(voc, run_time)
        plt.plot(voc, run_time_pred, c = 'red')
        plt.title(f'{alg}_algorithm_log – log(vocab) vs. running time (sec)')
        plt.xlabel('log(Vocabulary)') 
        plt.ylabel('Running time (sec)')
        plt.savefig(graph_name)

    def gen_graph_exp(self, alg, graph_name):   
        voc, _, run_time, run_time_pred, _ = self.graph_analysis_log(alg)
        plt.figure(figsize = (10,6))
        plt.scatter(voc, run_time)
        plt.plot(voc, run_time_pred, c = 'red')
        plt.title(f'{alg}_algorithm_exp – vocab vs. log(running time)')
        plt.xlabel('Vocabulary') 
        plt.ylabel('log(Running time)')
        plt.savefig(graph_name)


    max_poly_degree = 4 
    #datafile = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_DRAFT.csv'
    #datafile = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_FINAL.csv'
    def graph_results_poly_dp(self, alg, max_poly_degree):
        for deg in range(1, max_poly_degree+1):
            graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\dp_graph_poly_{deg}.png'
            GraphAnalysis(self.datafile).gen_graph_poly(alg, deg, graph_name)
            print(f'dp results for polynomial of degree {deg}:')
            _, Voc_poly, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_poly(alg, deg)
            print(f'dp_poly_{deg} R2: {model.score(Voc_poly, run_time)}')
            print(f'dp_poly_{deg} coefficients: {model.estimator_.coef_}')
            print(f'dp_poly_{deg} intercept: {model.estimator_.intercept_}\n')

    def graph_results_log_dp(self, alg):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\dp_graph_log.png' 
        GraphAnalysis(self.datafile).gen_graph_log(alg, graph_name)
        print(f'dp results for log graph:')
        _, Voc_poly_log, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_log(alg)
        print(f'dp_log R2: {model.score(Voc_poly_log, run_time)}')
        print(f'dp_log coefficients: {model.estimator_.coef_}')
        print(f'dp_log intercept: {model.estimator_.intercept_}\n')
    
    def graph_results_exp_dp(self, alg):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\dp_graph_exp.png' 
        GraphAnalysis(self.datafile).gen_graph_exp(alg, graph_name)
        print(f'dp results for exp graph:')
        _, Voc_poly, run_time_log,_, model = GraphAnalysis(self.datafile).graph_analysis_exp(alg)
        print(f'dp_exp R2: {model.score(Voc_poly, run_time_log)}')
        print(f'dp_exp coefficients: {model.estimator_.coef_}')
        print(f'dp_exp intercept: {model.estimator_.intercept_}\n')
    
    def graph_results_poly_sk(self, alg, max_poly_degree):
        for deg in range(1, max_poly_degree+1):
            graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\sk_graph_poly_{deg}.png'
            GraphAnalysis(self.datafile).gen_graph_poly(alg, deg, graph_name)
            print(f'sk results for polynomial of degree {deg}:')
            _, Voc_poly, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_poly(alg, deg)
            print(f'sk_poly_{deg} R2: {model.score(Voc_poly, run_time)}')
            print(f'sk_poly_{deg} coefficients: {model.estimator_.coef_}')
            print(f'sk_poly_{deg} intercept: {model.estimator_.intercept_}\n')

    def graph_results_log_sk(self, alg):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\sk_graph_log.png' 
        GraphAnalysis(self.datafile).gen_graph_log(alg, graph_name)
        print(f'sk results for log graph:')
        _, Voc_poly_log, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_log(alg)
        print(f'sk_log R2: {model.score(Voc_poly_log, run_time)}')
        print(f'sk_log coefficients: {model.estimator_.coef_}')
        print(f'sk_log intercept: {model.estimator_.intercept_}\n')
    
    def graph_results_exp_sk(self, alg):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\sk_graph_exp.png' 
        GraphAnalysis(self.datafile).gen_graph_exp(alg, graph_name)
        print(f'sk results for exp graph:')
        _, Voc_poly, run_time_log,_, model = GraphAnalysis(self.datafile).graph_analysis_exp(alg)
        print(f'sk_exp R2: {model.score(Voc_poly, run_time_log)}')
        print(f'sk_exp coefficients: {model.estimator_.coef_}')
        print(f'sk_exp intercept: {model.estimator_.intercept_}\n')
        
    def graph_results_poly_st(self, alg, max_poly_degree):
        for deg in range(1, max_poly_degree+1):
            graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\st_graph_poly_{deg}.png'
            GraphAnalysis(self.datafile).gen_graph_poly(alg, deg, graph_name)
            print(f'st results for polynomial of degree {deg}:')
            _, Voc_poly, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_poly(alg, deg)
            print(f'st_poly_{deg} R2: {model.score(Voc_poly, run_time)}')
            print(f'st_poly_{deg} coefficients: {model.estimator_.coef_}')
            print(f'st_poly_{deg} intercept: {model.estimator_.intercept_}\n')

    def graph_results_log_st(self, alg):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\st_graph_log.png' 
        GraphAnalysis(self.datafile).gen_graph_log(alg, graph_name)
        print(f'st results for log graph:')
        _, Voc_poly_log, run_time,_, model = GraphAnalysis(self.datafile).graph_analysis_log(alg)
        print(f'st_log R2: {model.score(Voc_poly_log, run_time)}')
        print(f'st_log coefficients: {model.estimator_.coef_}')
        print(f'st_log intercept: {model.estimator_.intercept_}\n')

    def graph_results_exp_st(self, alg): 
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\st_graph_exp.png' 
        GraphAnalysis(self.datafile).gen_graph_exp(alg, graph_name)
        print(f'st results for exp graph:')
        _, Voc_poly, run_time_log,_, model = GraphAnalysis(self.datafile).graph_analysis_exp(alg)
        print(f'st_exp R2: {model.score(Voc_poly, run_time_log)}')
        print(f'st_exp coefficients: {model.estimator_.coef_}')
        print(f'st_exp intercept: {model.estimator_.intercept_}\n')

#if __name__ == '__main__':
    