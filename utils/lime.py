from lime.lime_text import LimeTextExplainer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline




class MyDNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dnn):
        self.dnn = dnn

    def fit(self, x, y):
        return self
    
    def transform(self, X):
        return self.dnn.predict(X)

    def predict_proba(self, X):
        # preds = np.zeros(shape=(len(X),2), dtype=float)
        predictions = self.dnn.predict(X)
        return predictions


class Lime():

    def __init__(self, _classifier):
        self.classifier = _classifier
        self.dnn_pipeline = make_pipeline(MyDNNTransformer(_classifier))
    


    def explain(self, data, labels):
        explainer = LimeTextExplainer()
        explanations = []
        for idx in range(0, len(data)):
            exp = explainer.explain_instance(data[idx], self.dnn_pipeline.predict_proba, num_features=2000)
            print(exp)
            explanations.append(exp)
        
        return explanations

