from sklearn.datasets import load_breast_cancer
import pandas as pd
import random
import numpy as np
from sklearn.metrics import confusion_matrix
#from multiprocess import Pool
#from itertools import product


class ExNode:
    def __init__(self, size, left=None, right=None):
        self.size=size
        self.left=left
        self.right=right


class InNode:
    def __init__(self, p, q, left=None, right=None):
        self.left = left
        self.right = right
        self.p = p
        self.q = q


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        #self.trees = []        
        if isinstance(X, pd.DataFrame):
            X = X.values
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            X_sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            itree = IsolationTree(height_limit)
            itree.fit(X_sample, improved) 
            self.trees.append(itree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        def path_length_per_x(x, tree, e=0):
            while isinstance(tree, InNode):
                a = tree.q
                if x[a] < tree.p:
                    tree = tree.left
                    e+=1
                else:
                    tree = tree.right
                    e+=1

            if tree.size > 2:
                c = 2*(np.log(tree.size-1)+0.5772156649)-2*(tree.size-1)/tree.size
            elif tree.size==2:
                c = 1
            else: c = 0
            return e+c

        e=0  # initialize current path       
        path_lengths = []
        for x in X:
            for tree in self.trees:
                path_lengths.append(path_length_per_x(x, tree.root))

        avgPathLengths = np.array(path_lengths).reshape(len(X), -1)
        avgPathLengths = np.mean(avgPathLengths, axis=1)
        
        # why multiprocess doesn't run faster?
        # p = Pool(4)
        # list_root = [tree.root for tree in self.trees]
        # myInput = product(X, list_root)
        # pathLengths = p.starmap(path_length_per_x, myInput)
        # pathLengths = np.array(pathLengths).reshape(len(X), -1)
        # avgPathLengths = np.mean(pathLengths, axis=1)
        return avgPathLengths
        
    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if self.sample_size > 2:
            c = 2*(np.log(self.sample_size-1)+0.5772156649)-2*(self.sample_size-1)/self.sample_size
        elif self.sample_size==2:
            c = 1
        else: c = 0

        H = self.path_length(X)
        score = 2**(-1*H/c)
        #print(score)
        return score

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return [1 if s>=threshold else 0 for s in scores]

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        pred = self.predict_from_anomaly_scores(scores, threshold)
        return pred


class IsolationTree:
    def __init__(self, height_limit, current_height=0, n_nodes=0, p=None, q=None):
        self.height_limit = height_limit
        #self.max_height = np.floor(height_limit)
        #self.X = X
        self.root=InNode(p, q)
        self.current_height=current_height
        self.n_nodes = n_nodes
        self.p=p
        self.q=q

    def fit(self, X:np.ndarray, improved):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if improved == False:
            if (self.current_height >= self.height_limit) or (len(X) <= 1):
                self.root=ExNode(len(X))
                return self.root
            else:
                Q = np.arange(X.shape[1])
                self.q = random.choice(Q)
                #self.p = random.uniform(X[:, self.q].min(), X[:, self.q].max())

                p_min = X[:, self.q].min()
                p_max = X[:, self.q].max()

                if p_min==p_max:
                	self.root=ExNode(len(X))
                	return self.root

                self.p = random.uniform(p_min, p_max)
                X_l = X[X[:, self.q]< self.p]
                X_r = X[X[:, self.q]>=self.p]


                l_tree = IsolationTree(self.height_limit, self.current_height+1).fit(X_l, improved)
                r_tree = IsolationTree(self.height_limit, self.current_height+1).fit(X_r, improved)
                #self.root = InNode(self.p, self.q, l_tree, r_tree)
                #self.root = InNode(self.p, self.q, self.fit(X_l, improved), self.fit(X_r, improved))
                #self.n_nodes = self.count_nodes(self.root)
                self.root = self.root = InNode(self.p, self.q, l_tree, r_tree)
                self.n_nodes = 1+self.count_nodes(l_tree)+ self.count_nodes(r_tree)
                
                return self.root
        
        else:
            if (self.current_height>=self.height_limit) or (len(X)<=1):
                self.root=ExNode(len(X))
                return self.root
            else:
                Q = np.arange(X.shape[1])
                q_list = np.random.randint(X.shape[1], size=4)
                #q_list = random.sample(list(Q), 4)

                dist_qp = dict()
                for q in q_list:
                	p = np.random.uniform(X[:, q].min(), X[:, q].max())
                	size = min(len(X[X[:, q]<p]), len(X[X[:, q]>p]))
                	dist_qp[size]=(q,p)
                tuple = dist_qp[min(dist_qp.keys())]
                self.q = tuple[0]
                self.p = tuple[1]

                # l_index = X[:, self.q]< self.p
                # r_index = np.invert(l_index)
                # X_l = X[l_index]
                # X_r = X[r_index]
                if X[:, self.q].min()==X[:, self.q].max():
                	self.root=ExNode(len(X))
                	return self.root

                X_l = X[X[:, self.q]< self.p]
                X_r = X[X[:, self.q]>=self.p]
                l_tree = IsolationTree(self.height_limit, self.current_height+1).fit(X_l, improved)
                r_tree = IsolationTree(self.height_limit, self.current_height+1).fit(X_r, improved)
                self.root = InNode(self.p, self.q, l_tree, r_tree)
                self.n_nodes = 1+self.count_nodes(l_tree)+ self.count_nodes(r_tree)
                return self.root
    
    def count_nodes(self, node):
        if node is None: return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1
    #TPR = -1
    while threshold>0:
        y_hat = [1 if s>=threshold else 0 for s in scores]
        confusion = confusion_matrix(y, y_hat)
        TN, FP, FN, TP = confusion.flat
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        if TPR < desired_TPR:
        	threshold -= 0.01
        else: break

    return threshold, FPR

