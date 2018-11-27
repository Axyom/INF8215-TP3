from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math as mt

class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        


    def fit(self, X, y=None):
        
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))
        

        X_bias = np.c_[ np.ones(X.shape[0]), X ]        
        self.theta_ = np.random.rand(self.nb_feature + 1, self.nb_classes)
        

#         for epoch in range( self.n_epochs):
#             #Je calcule les logits

#             logits = X_bias @ self.theta_  
#             #Je transforme en proba avec softmax
#             p = self.predict_proba(X,y)

#             loss = self._cost_function(p, y) 

#             self.theta_ = self.theta_ - self.lr * self._get_gradient(X_bias,y, p)

            if (len(self.losses_) > 0 ):
                diff = self.losses_[-1]-loss
                self.losses_.append(loss)
                if (abs(diff) < self.threshold):
                    return self
            else:
                self.losses_.append(loss)
                
        return self

    
    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
#         print("X")
#         print(X)
#         print("Theta")
#         print(self.theta_)
        X_bias = np.c_[ np.ones(X.shape[0]), X ]
#         print("X biais")
#         print(X_bias)
        z = np.dot(X_bias, self.theta_)
#         print("Z")
#         print(z)
        p = self._softmax(z)
#         print("Softmax")
#         print(p)
        return p

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        temp = np.unique(y)
        X_bias = np.c_[ np.ones(X.shape[0]), X ]
        p = self.predict_proba(X, y)
        predictions = p.argmax(axis=1)
        return predictions #self._one_hot(predictions)
        

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)   

    def score(self, X, y=None):
        probabilities = self.predict(X)
        return self._cost_function(probabilities, y )

    def _cost_function(self,probabilities, y ): 
        m = probabilities.shape[0]
        # One-hot encode y
        y_oh = self._one_hot(y)
        p = np.clip(probabilities, self.eps, 1-self.eps)
        J = 0     
        J = - np.sum(y_oh*np.log(p))/ m
#         for i in range(y_oh.shape[0]):
#             for k in range(y_oh.shape[1]):
#                 J = J + y_oh[k][i]*mt.log(p[k][i])
#         J= -J/m
        if self.regularization:
            l2 = 0
#             for i in range(self.nb_feature):
#                 for k in range(1,self.nb_classes):
#                     l2 = l2 + self.theta_[k,i]**2
#             l2 = l2 * self.alpha
            l2= (self.alpha * np.sum(self.theta_[1:,:]**2))
            J = J + l2
        
        return J

    def _one_hot(self,y):
        temp = np.unique(y)
        oneHot = np.zeros((len(y), self.nb_classes))
        j=0
        for i in y:
            index = list(temp).index(i)
            oneHot[j][index]=1
            j += 1
        return oneHot

    def _softmax(self,z):
        SM = np.empty(z.shape)
        temp = np.amax(z, axis = 1)
        for i in range(z.shape[0]):
            SM[i,:]  = np.exp(z[i,:] - temp[i])
            SM[i,:] /= np.sum(SM[i,:])
        return SM

    def _get_gradient(self,X_bias,y, probas):
        m = probas.shape[0]
        loss = probas - self._one_hot(y)
        xTrans = X_bias.transpose()
        # Delta : number of features + 1 * number of classes 
        delta = np.dot(xTrans, loss)
        if self.regularization:
            delta[1:,:] = delta[1:,:] + self.alpha * self.theta_[1:,:]
        return delta/m
    
    