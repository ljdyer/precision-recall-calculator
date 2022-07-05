import pandas as pd
import numpy as np
from typing import Any, Union
from numbers import Number
from sklearn.metrics import ConfusionMatrixDisplay

Str_or_List = Union[str, list]
Float_or_Str = Union[float, str]
Int_or_Str = Union[int, str]


# ====================
def get_feature_positions(input_str: str, output_str: str, features: list) -> dict:
    """Get a list of booleans representing whether a feature applies to each
    position in output_str.

    E.g. for the output_str "I like bananas.' with features 'CAPITALISATION' and
    '.', the output would be:
    {
        'CAPITALISATION': [True, False, False, ..., False],
        '.': [False, False, False, ..., True]
    }"""

    positions = {f: [False] * len(input_str) for f in features}
    output_str_ = list(output_str)
    for i, char in enumerate(input_str):
        next_char = output_str_.pop(0)
        if 'CAPITALISATION' in features and next_char.isupper():
            positions['CAPITALISATION'][i] = True
            next_char = next_char.lower()
        assert next_char == char
        while len(output_str_) > 0 and output_str_[0] in features:
            feature_char = output_str_.pop(0)
            positions[feature_char][i] = True
    return positions


# ====================
def get_ref_and_hyp_feature_positions(ref: str, hyp: str, features: list):

    input_str = get_input_str(ref, features)
    feature_positions = {}
    feature_positions['ref'] = get_feature_positions(input_str, ref, features)
    feature_positions['hyp'] = get_feature_positions(input_str, hyp, features)
    return feature_positions

# ====================
def get_feature_positions(self, doc_idx: int):

    strings = {
        'ref': self.ref[doc_idx],
        'hyp': self.hyp[doc_idx]
    }
    features_present = {
        'ref': [],
        'hyp': []
    }
    while strings['ref'] and strings['hyp']:
        ref_char = strings['ref'].pop(0)
        hyp_char = strings['hyp'].pop(0)
        assert strings['ref'].lower() == strings['hyp'].lower()
        for string in strings.keys():
            if 'CAPITALISATION' in self.features and strings[string].isupper():
                features_present.append('CAPITALISATION')
            while strings[string][0] in self.features:
                features_present.append(strings[string].pop(0))
    feature_positions = {
        'ref': {f: [f in features_present['ref'] for f in self.features]},
        'hyp': {f: [f in features_present['hyp'] for f in self.features]},
    }
    return feature_positions
            








# ====================
def str_or_list_to_list(str_or_list: Str_or_List) -> list:

    if isinstance(str_or_list, str):
        return [str_or_list]
    else:
        return str_or_list


# ====================
class PrecisionRecallGetter:

    # ====================
    def __init__(self, 
                 reference: Str_or_List,
                 hypothesis: Str_or_List,
                 capitalisation: bool,
                 feature_chars: Str_or_List):

        self.capitalisation = capitalisation
        
        self.reference = str_or_list_to_list(reference)
        self.hypothesis = str_or_list_to_list(hypothesis)
        if len(self.reference) != len(self.hypothesis):
            raise ValueError("Hypothesis and reference lists must have equal length.")

        if isinstance(feature_chars, str):
            feature_chars = list(feature_chars)
        else:
            feature_chars = feature_chars
        self.feature_chars = feature_chars

        if capitalisation:
            features = feature_chars.copy() + ['CAPITALISATION']
        else:
            features = feature_chars.copy()
        self.features = features

        self.feature_positions = []
        self.precision_recall_fscore = []
        for doc_idx in range(len(self.hypothesis)):
            self.feature_positions.append(
                self.get_feature_positions_doc(doc_idx))
        
        




    # ====================
    def get_feature_positions_doc(self, doc_idx: int):

        strings = {
            'ref': list(self.reference[doc_idx]),
            'hyp': list(self.hypothesis[doc_idx])
        }
        features_present = {
            'ref': [],
            'hyp': []
        }
        while strings['ref'] and strings['hyp']:
            next_char = {
                'hyp': strings['ref'].pop(0),
                'ref': strings['hyp'].pop(0)
            }
            assert next_char['ref'].lower() == next_char['hyp'].lower()
            for string in strings.keys():
                features_present[string].append([])
                if 'CAPITALISATION' in self.features and next_char[string].isupper():
                    features_present[string][-1].append('CAPITALISATION')
                while (len(strings[string]) > 0
                       and strings[string][0] in self.features):
                    features_present[string][-1].append(strings[string].pop(0))
        print(features_present)
        feature_positions = {
            'ref': {f: [f in x for x in features_present['ref']] for f in self.features},
            'hyp': {f: [f in x for x in features_present['hyp']]  for f in self.features},
        }
        return feature_positions

    # ====================
    def show_confusion_matrix(self, doc_idx: Int_or_Str = 'all', feature: str = 'all'):

        if isinstance(doc_idx, int):
            ref = self.feature_positions[doc_idx]['ref'][feature]
            hyp = self.feature_positions[doc_idx]['hyp'][feature]
        else:
            raise RuntimeError('"all" docs option not implemented yet!')

        ConfusionMatrixDisplay.from_predictions(ref, hyp)
        

    # ====================
    @staticmethod
    def feature_display_name(feature):
        pass

# # ====================
# class DocInfo:

#     # ====================
#     def __init__(self, 
#                  ref: str = None,
#                  hyp: str = None,
#                  features: list = None):

#         self.ref = ref
#         self.hyp = hyp
#         self.features = features
#         if hyp:
#             self.stripped = self.strip_features(hyp)
#         self.feature_positions_hyp = self.get_feature_positions(self.hyp)
#         self.feature_positions_ref = self.get_feature_positions(self.ref)

#     # ====================
#     def strip_features(self, str_: str):

#         if 'CAPITALISATION' in self.features:
#             str_ = str_.lower()
#         for feature_char in self.features:
#             if feature_char != 'CAPITALISATION':
#                 input_str = input_str.replace(feature_char, '')
#         return str_

#     # ====================
#     def get_feature_positions(self, str_: str):

#         positions = {
#             f: [False] * len(self.stripped) for f in self.features}
#         str_ = list(str_)
#         for i, char in enumerate(self.stripped):
#             next_char = str_.pop(0)
#         if 'CAPITALISATION' in self.features and next_char.isupper():
#             positions['CAPITALISATION'][i] = True
#             next_char = next_char.lower()
#         try:
#             assert next_char == char
#         except AssertionError:
#             raise ValueError('Character mismatch in position {i}!')
#         while len(str_) > 0 and str_[0] in self.features:
#             feature_char = str_.pop(0)
#             positions[feature_char][i] = True
#         return positions

#     # ====================
#     def __add__(self, other):

#         result = {}
#         for feature in self.features:
#             result['hyp'][feature] = 







if __name__ == "__main__":

    REFERENCE = ["I like bananas.", 'HAPPY BIRTHDAY', 'SeNTEncE3']
    HYPOTHESIS = ["i like bananas...", 'happy. birthday,', 'SeNTEncE3']
    CAPITALISATION = True
    FEATURE_CHARS = list(' ,.')
    prg = PrecisionRecallGetter(reference=REFERENCE,
                                hypothesis=HYPOTHESIS,
                                capitalisation=CAPITALISATION,
                                feature_chars=FEATURE_CHARS)
    # print(prg.get_feature_positions(0))
    prg.show_confusion_matrix(1, 'CAPITALISATION')

