
import alibi.explainers as alibiexp
import tensorflow as tf
from config.config_featuremaps import INPUT_MAXLEN, VOCAB_SIZE
from feature_map.imdb.predictor import Predictor
from utils.text.processor import process_text_contributions

class IntegratedGradients:

    def __init__(self, _classifier):
        self.explainer = alibiexp.IntegratedGradients(_classifier, layer=_classifier.layers[2], n_steps=50, method="gausslegendre", internal_batch_size=100)
    
    def explain(self, data, labels):
        explanation = self.explainer.explain(data, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
        attrs = explanation.attributions[0]
        contributions = attrs.sum(axis=2)
        contributions_processed = process_text_contributions(data, contributions)
        return contributions_processed


    def export_explanation(self, data, labels, file_name):

        seq = Predictor.tokenizer.texts_to_sequences(data)
        data = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=INPUT_MAXLEN)

        explanation = self.explainer.explain(data, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
        attrs = explanation.attributions[0]
        contributions = attrs.sum(axis=2)
        expl = contributions[0]

        text = Predictor.tokenizer.sequences_to_texts(seq)
        text = text[0].split()

        first_index = VOCAB_SIZE - len(text)
        colors = colorize(expl[first_index:])

        _data = HTML("".join(list(map(hlstr, text, colors))))

        with open(f"{file_name}.html", "w") as file:
            file.write(_data.data)

from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"



def colorize(attrs, cmap='Reds'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = attrs.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors