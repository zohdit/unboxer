
from lime.lime_text import LimeTextExplainer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.text.processor import process_text_contributions
from config.config_data import INPUT_MAXLEN
from feature_map.imdb.predictor import Predictor


class MyDNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dnn):
        self.dnn = dnn

    def fit(self, x, y):
        return self
    
    def transform(self, X):
        return self.dnn.predict(X)

    def predict_proba(self, X):
        seq = Predictor.tokenizer.texts_to_sequences(X)
        padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
        predictions = self.dnn.predict(padded_texts)
        return predictions


class Lime():

    def __init__(self, _classifier):
        self.explainer = LimeTextExplainer(class_names=[0,1])
        self.classifier = _classifier
        self.dnn_pipeline = make_pipeline(MyDNNTransformer(_classifier))
    
    def explain(self, data, labels):
        explanations = [] 
        seq = Predictor.tokenizer.texts_to_sequences(data)
        texts = Predictor.tokenizer.sequences_to_texts(seq)       
        for idx in range(len(texts)):
            explanation = self.explainer.explain_instance(data[idx], self.dnn_pipeline.predict_proba, num_features=1000)
            contribution = []
            
            exp = explanation.as_list()
            for ss in seq[idx]:
                flag = False
                word = Predictor.tokenizer.sequences_to_texts([[ss]])[0]
                for pair in exp:
                    if pair[0].casefold() == word.casefold():
                        contribution.append(pair[1])
                        flag = True
                        break
                if flag == False:
                    contribution.append(0)            
            explanations.append(contribution)
               
        contributions_processed = process_text_contributions(seq, explanations)
        return contributions_processed

    def export_explanation(self, data, labels, file_name):     
        for idx in range(len(data)):
            explanation = self.explainer.explain_instance(data[idx], self.dnn_pipeline.predict_proba, num_features=20)

            with open(f'{file_name}.html', "w") as file:
                explanation.save_to_file(f'{file_name}.html')
        
        return explanation.as_list()

