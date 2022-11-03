

from utils.text.processor import process_text_contributions

class GradCAM:

    def __init__(self, _classifier):
        self.explainer = None


    def explain(self, data, labels):
        explanation = self.explainer.explain(data, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
        attrs = explanation.attributions[0]
        contributions = attrs.sum(axis=2)
        contributions_processed = process_text_contributions(data, contributions)
        return contributions_processed