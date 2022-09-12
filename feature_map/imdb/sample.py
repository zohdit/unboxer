import json
from os.path import join

from feature_map.imdb.feature_simulator import FeatureSimulator



class Sample:
    def __init__(self, text, label, prediction):
        self.id = id(self)
        self.text = text
        self.expected_label = label
        self.predicted_label = prediction
        self.features = {
            feature_name: feature_simulator(self)
            for feature_name, feature_simulator in FeatureSimulator.get_simulators().items()
        }

    def to_dict(self):
        return {'id': id(self),
                'text': self.text,
                'expected_label': self.expected_label,
                'predicted_label': self.predicted_label,
                'misbehaviour': self.is_misbehavior,
                'features': self.features,
                }

    @property
    def is_misbehavior(self):
        return self.expected_label != self.predicted_label

    def from_dict(self, the_dict):
        for k in self.__dict__.keys():
            if k in the_dict.keys():
                setattr(self, k, the_dict[k])
        return self

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename + ".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))


    def export(self, dst):
        dst = join(dst, "mbr" + str(self.id))
        self.dump(dst)

