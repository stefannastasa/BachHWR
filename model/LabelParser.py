from typing import List, Sequence


class LabelParser:
    def __init__(self):
        self.classes = None
        self.vocab_size = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.ctc_classes = None
        self.ctc_idx_to_class = None
        self.ctc_class_to_idx = None

    def fit(self, classes: Sequence[str]):
        self.classes = list(classes)
        self.vocab_size = len(classes)
        self.idx_to_class = dict(enumerate(classes))
        self.class_to_idx = {cls: i for i, cls in self.idx_to_class.items()}

        self.ctc_classes = ["<blank>"] + self.classes
        self.ctc_idx_to_class = dict(enumerate(self.ctc_classes))
        self.ctc_class_to_idx = {cls: i for i, cls in self.ctc_idx_to_class.items()}

        return self

    def addClasses(self, classes: List[str]):
        all_classes = sorted(set(self.classes + classes))

        self.fit(all_classes)

    def encode_labels(self, sequence: Sequence[str]):
        self._check_fitted()
        return [self.class_to_idx[c] for c in sequence]

    def decode_labels(self, sequence: Sequence[int]):
        self._check_fitted()
        return [self.idx_to_class[c] for c in sequence]

    def ctc_encode_labels(self, sequence: Sequence[str]):
        self._check_fitted()
        return [self.ctc_class_to_idx[c] for c in sequence]

    def ctc_decode_labels(self, sequence: Sequence[int]):
        self._check_fitted()
        classes = [self.ctc_idx_to_class[c] for c in sequence]
        result = []
        for i, char in enumerate(classes):
          if i == 0 or char != classes[i-1]:
              result.append(char)
        return result

    def _check_fitted(self):
        if self.classes is None:
            raise ValueError("LabelParser class was not fitted yet")