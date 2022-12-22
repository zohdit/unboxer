N
import alibi.explainers as alibiexp
from config.config_data import INPUT_MAXLEN
import tensorflow as tf
from feature_map.imdb.predictor import Predictor
from utils.text.processor import process_text_contributions

class IntegratedGradients:

    def __init__(self, _classifier):
        self.explainer = alibiexp.IntegratedGradients(_classifier, layer=_classifier.layers[1], n_steps=50, method="gausslegendre", internal_batch_size=100)
    
    def explain(self, data, labels):
        explanation = self.explainer.explain(data, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
        attrs = explanation.attributions[0]        
        contributions = attrs.sum(axis=2)
        contributions_processed = process_text_contributions(data, contributions)
        return contributions_processed


    def export_explanation(self, data, labels, file_name):

        seq = Predictor.tokenizer.texts_to_sequences(data)
        data_padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=INPUT_MAXLEN)

        explanation = self.explainer.explain(data_padded, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
        attrs = explanation.attributions[0]

        contributions = attrs.sum(axis=2)
        expl = contributions[0]

        
        text = Predictor.tokenizer.sequences_to_texts(seq)

        text = text[0].split()  
        first_index = INPUT_MAXLEN - len(text)      

        list_words = []

        expl = expl[first_index:]

        # iterate on processed contribution which is a vector with INPUT_MAXLEN size
        for idx1 in range(len(expl)):
            # check if the word has contribution != 0
            if abs(expl[idx1]) > 0:
                # find the corresponding word and add it to the list of important words if its not already added
                word = text[idx1]
                list_words.append([word, expl[idx1]])

        colors = colorize(expl)

        _data = HTML("".join(list(map(hlstr, text, colors))))

        with open(f"{file_name}.html", "w") as file:
            file.write(_data.data)

        return list_words

from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"



def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = attrs.max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors