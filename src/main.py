import pandas as pd
from typing import Union
from sklearn.metrics import confusion_matrix
from helper import display_or_print, get_tqdm
import numpy as np

Str_or_List = Union[str, list]
Int_or_Str = Union[int, str]

NON_EQUAL_LENGTH_ERROR = \
    "Hypothesis and reference lists must have equal length."
DIFFERENT_CHARS_ERROR = """
    Different characters found between reference and hypothesis strings in \
document index: {doc_idx}!
    Reference: {ref_str}
    Hypothesis: {hyp_str}
    """
INIT_COMPLETE_MSG = "Initialisation complete."


tqdm_ = get_tqdm()

FEATURE_DISPLAY_NAMES = {
    'CAPITALISATION': "Capitalisation",
    ' ': "Spaces (' ')",
    ',': "Commas (',')",
    '.': "Periods ('.')",
    'all': 'All features'
}


# ====================
def str_or_list_to_list(str_or_list: Str_or_List) -> list:

    if isinstance(str_or_list, str):
        return [str_or_list]
    else:
        return str_or_list


# ====================
class PrecisionRecallCalculator:

    # ====================
    def __init__(self,
                 reference: Str_or_List,
                 hypothesis: Str_or_List,
                 capitalisation: bool,
                 feature_chars: Str_or_List,
                 get_cms_on_init: bool = True):

        self.reference = str_or_list_to_list(reference)
        self.hypothesis = str_or_list_to_list(hypothesis)
        if len(self.reference) != len(self.hypothesis):
            raise ValueError(NON_EQUAL_LENGTH_ERROR)
        self.set_feature_chars(feature_chars)
        self.set_features(capitalisation)
        if get_cms_on_init:
            self.get_confusion_matrices()
            self.get_confusion_matrix_all()

        print(INIT_COMPLETE_MSG)

    # ====================
    def set_feature_chars(self, feature_chars: Str_or_List):
        """Set self.feature_chars

        If feature_chars is provided as a string, convert it to a list of
        characters."""

        if isinstance(feature_chars, str):
            self.feature_chars = list(feature_chars)
        else:
            self.feature_chars = feature_chars

    # ====================
    def set_features(self, capitalisation: bool):
        """Set self.features

        If capitalisation=True, add 'CAPITALISATION' to the list of
        feature_chars."""

        if capitalisation:
            self.features = \
                self.feature_chars.copy() + ['CAPITALISATION']
        else:
            self.features = self.feature_chars.copy()

    # ====================
    def get_confusion_matrices(self):
        """Get confusion matrices for all documents in the corpus."""

        print("Getting confusion matrices...")
        self.confusion_matrices = {}
        for doc_idx in tqdm_(range(len(self.hypothesis))):
            self.confusion_matrices[doc_idx] = (
                self.get_confusion_matrices_doc(doc_idx))

    # ====================
    def get_confusion_matrices_doc(self, doc_idx: int):
        """Get confusion matrics for a single document."""

        strings = {
            'ref': list(self.reference[doc_idx].strip()),
            'hyp': list(self.hypothesis[doc_idx].strip())
        }
        features_present = {'ref': [], 'hyp': []}
        while strings['ref'] and strings['hyp']:
            next_char = {
                'hyp': strings['ref'].pop(0),
                'ref': strings['hyp'].pop(0)
            }
            try:
                assert next_char['ref'].lower() == next_char['hyp'].lower()
            except AssertionError:
                error_msg = DIFFERENT_CHARS_ERROR.format(
                    doc_idx=doc_idx,
                    ref_str=(next_char['ref'] + ''.join(strings['ref'][:10])),
                    hyp_str=(next_char['hyp'] + ''.join(strings['hyp'][:10]))
                )
                raise ValueError(error_msg)
            for string in strings.keys():
                features_present[string].append([])
                if ('CAPITALISATION' in self.features
                   and next_char[string].isupper()):
                    features_present[string][-1].append('CAPITALISATION')
                while (len(strings[string]) > 0
                       and strings[string][0] in self.features):
                    features_present[string][-1].append(strings[string].pop(0))
        confusion_matrices = {
            f: confusion_matrix(
                [f in x for x in features_present['ref']],
                [f in x for x in features_present['hyp']],
                labels=[True, False]
            )
            for f in self.features
        }
        confusion_matrix_all = sum(
            confusion_matrices[f] for f in self.features)
        confusion_matrices['all'] = confusion_matrix_all
        return confusion_matrices

    # ====================
    def get_confusion_matrix_all(self):
        """Get the confusion matrix for the entire corpus."""

        all_docs = {}
        for f in self.features + ['all']:
            all_docs[f] = \
                sum([self.confusion_matrices[doc_idx][f]
                    for doc_idx in range(len(self.reference))])
        self.confusion_matrices['all'] = all_docs

    # ====================
    def show_confusion_matrices(self, doc_idx: Int_or_Str = 'all'):
        """Show confusion matrices for each feature, for either a
        single document or the entire corpus."""

        for feature in self.features + ['all']:
            print(self.feature_display_name(feature))
            print()
            cm = self.confusion_matrices[doc_idx][feature]
            col_index = pd.MultiIndex.from_tuples(
                [('Hypothesis', 'positive'), ('Hypothesis', 'negative')])
            row_index = pd.MultiIndex.from_tuples(
                [('Reference', 'positive'), ('Reference', 'negative')])
            display_or_print(pd.DataFrame(
                cm, index=row_index, columns=col_index))
            print()

    # ====================
    def show_precision_recall_fscore(self, doc_idx: Int_or_Str = 'all'):
        """Show precision, recall and F-score for each feature, for
        either a single document or the entire corpus."""

        feature_scores = {
            self.feature_display_name(feature):
            self.precision_recall_fscore_from_cm(
                self.confusion_matrices[doc_idx][feature])
            for feature in self.features + ['all']}
        display_or_print(pd.DataFrame(feature_scores).transpose())

    # ====================
    def precision_recall_fscore_from_cm(self, cm: np.ndarray):
        """Calculate precision, recall, and F-score from a confusion matrix."""

        true_pos = float(cm[0][0])
        false_pos = float(cm[1][0])
        false_neg = float(cm[0][1])
        try:
            precision = true_pos / (true_pos + false_pos)
        except ZeroDivisionError:
            precision = 'N/A'
        try:
            recall = true_pos / (true_pos + false_neg)
        except ZeroDivisionError:
            recall = 'N/A'
        try:
            fscore = (2*precision*recall) / (precision+recall)
        except (TypeError, ZeroDivisionError):
            fscore = 'N/A'
        return {
            'Precision': precision,
            'Recall': recall,
            'F-score': fscore
        }

    # ====================
    @staticmethod
    def feature_display_name(feature):
        """Return the display name for a feature."""

        if feature in FEATURE_DISPLAY_NAMES:
            return FEATURE_DISPLAY_NAMES[feature]
        else:
            return f"'{feature}'"
