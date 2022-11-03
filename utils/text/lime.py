
from config.config_data import EXPECTED_LABEL
import numpy as np

from lime.lime_text import LimeTextExplainer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.text.processor import process_text_contributions
from config.config_featuremaps import INPUT_MAXLEN
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
        for idx in range(len(data)):
            explanation = self.explainer.explain_instance(data[idx], self.dnn_pipeline.predict_proba, num_features=2000)
            contribution = []
            seq = Predictor.tokenizer.texts_to_sequences([data[idx]])
            text = Predictor.tokenizer.sequences_to_texts(seq)

            
            exp = explanation.as_list()

            # these explanation works only for misbehaviours (because considers the exp for the other class than expected class)
            if EXPECTED_LABEL == 1:
                for word in text[0].split():
                    flag = False
                    for pair in exp:
                        if pair[0] == word:
                            if pair[1] < 0:
                                contribution.append(abs(pair[1]))
                                flag = True
                                break
                    if flag == False:
                        contribution.append(0)


            if EXPECTED_LABEL == 0:
                for word in text[0].split():
                    flag = False
                    for pair in exp:
                        if pair[0] == word:
                            if pair[1] > 0:
                                contribution.append(abs(pair[1]))
                                flag = True
                                break
                    if flag == False:
                        contribution.append(0)
            
            explanations.append(contribution)
            
        seq = Predictor.tokenizer.texts_to_sequences(data)     
        contributions_processed = process_text_contributions(seq, explanations)
        return contributions_processed

    def export_explanation(self, data, labels, file_name):     
        for idx in range(len(data)):
            explanation = self.explainer.explain_instance(data[idx], self.dnn_pipeline.predict_proba, num_features=2000)
            
            seq = Predictor.tokenizer.texts_to_sequences([data[idx]])


            with open(f'{file_name}.html', "w") as file:
                explanation.save_to_file(f'{file_name}.html')


