"""
main.py

Main module for PrecisionRecallCalculator class
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from prc_helper import (Int_or_Str, Str_or_List, Str_or_List_or_Series,
                        display_or_print, get_tqdm)

NON_EQUAL_LENGTH_ERROR = \
    "Hypothesis and reference lists must have equal length."
WARNING_DIFFERENT_CHARS = """Different characters found between reference and \
hypothesis strings in document index: {doc_idx}! \
(Reference: "{ref_str}"; Hypothesis: "{hyp_str}"). \
Skipping this document (returning None)."""
INIT_COMPLETE_MSG = "Initialisation complete."
REF_OR_HYP_TYPE_ERROR = """
reference and hypothesis parameters must have type list, str, \
or pandas.Series"""

FEATURE_DISPLAY_NAMES = {
    'CAPITALISATION': "Capitalisation",
    ' ': "Spaces (' ')",
    ',': "Commas (',')",
    '.': "Periods ('.')",
    'all': 'All features'
}

tqdm_ = get_tqdm()


# ====================
def str_or_list_or_series_to_list(
     input_: Str_or_List_or_Series) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, pd.Series):
        return input_.to_list()
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(REF_OR_HYP_TYPE_ERROR)


# ====================
class PrecisionRecallCalculator:

    # ====================
    def __init__(self,
                 reference: Str_or_List_or_Series,
                 hypothesis: Str_or_List_or_Series,
                 capitalisation: bool,
                 feature_chars: Str_or_List,
                 get_cms_on_init: bool = True):
        """
        Initialize an instance of the PrecisionRecallCalculator class

        Required arguments:
        -------------------
        reference:                  Either a single string, or a list or
            Str_or_List_or_Series   pandas.Series object of strings
                                    ('documents') to use as the reference
                                    corpus.
        hypothesis:                 Either a single string, or a list or
            Str_or_List_or_Series   pandas.Series object of strings
                                    ('documents') to use as the hypothesis
                                    corpus.
                                    (Number of documents must be the same
                                    as reference.)
        capitalisation: bool        Whether or not to treat capitalisation
                                    as a feature to be assessed.
        feature_chars:              A string or list of characters containing
            Str_or_List             other characters to treat as features
                                    (e.g. '., ' for periods, commas, and
                                    spaces.)

        Optional keyword arguments:
        ---------------------------
        get_cms_on_init: bool       Whether or not to get confusion matrices
                                    for all reference/hypothesis documents
                                    on intiialization. Set to false and access
                                    manually to save time if only looking at
                                    metrics for a subset of documents in a
                                    large corpus.
        """

        self.reference = str_or_list_or_series_to_list(reference)
        self.hypothesis = str_or_list_or_series_to_list(hypothesis)
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
        """Get confusion matrices for a single document."""

        strings = {
            'ref': list(self.reference[doc_idx].strip()),
            'hyp': list(self.hypothesis[doc_idx].strip())
        }
        features_present = {'ref': [], 'hyp': []}
        while strings['ref'] and strings['hyp']:
            next_char = {
                'ref': strings['ref'].pop(0),
                'hyp': strings['hyp'].pop(0)
            }
            try:
                assert next_char['ref'].lower() == next_char['hyp'].lower()
            except AssertionError:
                error_msg = WARNING_DIFFERENT_CHARS.format(
                    doc_idx=doc_idx,
                    ref_str=(next_char['ref'] + ''.join(strings['ref'][:10])),
                    hyp_str=(next_char['hyp'] + ''.join(strings['hyp'][:10]))
                )
                print(error_msg)
                return None
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
                    for doc_idx in range(len(self.reference))
                    if self.confusion_matrices[doc_idx] is not None])
        self.confusion_matrices['all'] = all_docs

    # ====================
    def show_confusion_matrices(self, doc_idx: Int_or_Str = 'all'):
        """
        Show confusion matrices for each feature, for either a
        single document or the entire corpus.

        Optional keyword arguments:
        ---------------------------
        doc_idx: Int_or_Str         Either an integer indicating the index of
                                    the document to show confusion matrices
                                    for, or 'all' to show confusion matrices
                                    for all documents in the corpus (the
                                    default behaviour).
        """

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
        """
        Show precision, recall and F-score for each feature, for
        either a single document or the entire corpus.

        Optional keyword arguments:
        ---------------------------
        doc_idx: Int_or_Str         Either an integer indicating the index of
                                    the document to show metrics for, or 'all'
                                    to show metrics for all documents in the
                                    corpus (the default behaviour).
        """

        feature_scores = self.get_feature_scores(doc_idx)
        display_or_print(pd.DataFrame(feature_scores).transpose())

    # ====================
    def get_precision_recall_latex(self, doc_idx: Int_or_Str = 'all'):

        feature_scores = self.get_feature_scores(doc_idx)
        scores_df = pd.DataFrame(feature_scores).transpose()
        output_lines = []
        output_lines.append(r"\hline")
        output_lines.append(r"& \head{Precision} & \head{Recall} & " +
                            r"\head{F-score}\\")
        output_lines.append(r"\hline")
        for index, data in scores_df.iterrows():
            new_line = (f"{index} & {data['Precision']:.3f} & " +
                        f"{data['Recall']:.3f} & " +
                        f"{data['F-score']:.3f}\\")
            output_lines.append(new_line)
        return '\n'.join(output_lines)

    # ====================
    def get_feature_scores(self, doc_idx: Int_or_Str = 'all'):

        feature_scores = {
            self.feature_display_name(feature):
            self.precision_recall_fscore_from_cm(
                self.confusion_matrices[doc_idx][feature])
            for feature in self.features + ['all']
        }
        return feature_scores

    # ====================
    def precision_recall_fscore_from_cm(self, cm: np.ndarray):
        """Calculate precision, recall, and F-score from a confusion matrix."""

        tp = float(cm[0][0])
        tn = float(cm[1][1])
        fp = float(cm[1][0])
        fn = float(cm[0][1])
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 'N/A'
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 'N/A'
        try:
            fscore = (2*precision*recall) / (precision+recall)
        except (TypeError, ZeroDivisionError):
            fscore = 'N/A'
        try:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            accuracy = 'N/A'
        return {
            'Precision': precision,
            'Recall': recall,
            'F-score': fscore,
            'Accuracy': accuracy
        }

    # ====================
    def label_fps_and_fns(self, strings: str):

        output_chars = []
        while strings['ref'] and strings['hyp']:
            features_present = {'ref': [], 'hyp': []}
            next_char = {
                'ref': strings['ref'].pop(0),
                'hyp': strings['hyp'].pop(0)
            }
            try:
                assert next_char['ref'].lower() == next_char['hyp'].lower()
            except AssertionError:
                error_msg = WARNING_DIFFERENT_CHARS.format(
                    doc_idx="UNKNOWN",
                    ref_str=(next_char['ref'] + ''.join(strings['ref'][:10])),
                    hyp_str=(next_char['hyp'] + ''.join(strings['hyp'][:10]))
                )
                print(error_msg)
                return None
            for string in strings.keys():
                if ('CAPITALISATION' in self.features
                   and next_char[string].isupper()):
                    features_present[string].append('CAPITALISATION')
                while (len(strings[string]) > 0
                       and strings[string][0] in self.features):
                    features_present[string].append(strings[string].pop(0))
            if 'CAPITALISATION' in self.features:
                if ('CAPITALISATION' in features_present['ref']
                   and 'CAPITALISATION' not in features_present['hyp']):
                    output_chars.append(f"\\fn{{{next_char['hyp']}}}")
                elif ('CAPITALISATION' not in features_present['ref']
                        and 'CAPITALISATION' in features_present['hyp']):
                    output_chars.append(f"\\fp{{{next_char['hyp']}}}")
                else:
                    output_chars.append(next_char['hyp'])
            else:
                output_chars.append(next_char['hyp'])
            for feature in self.feature_chars:
                if (feature in features_present['ref']
                   and feature not in features_present['hyp']):
                    output_chars.append(f'\\fn{{\\mbox{{{feature}}}}}')
                elif (feature not in features_present['ref']
                        and feature in features_present['hyp']):
                    output_chars.append(f'\\fn{{\\mbox{{{feature}}}}}')
                elif (feature in features_present['ref']
                        and feature in features_present['hyp']):
                    output_chars.append(feature)
        return output_chars

    # ====================
    def latex_text_display(self, doc_idx: int, start_char: int = 0,
                           chars_per_row: int = 30, num_rows: int = 3):

        if start_char != 0:
            raise RuntimeError('Start char != 0 not implemented yet!')
        strings = {
            'ref': list(self.reference[doc_idx].strip()),
            'hyp': list(self.hypothesis[doc_idx].strip())
        }
        labelled = self.label_fps_and_fns(strings)
        rows = [[labelled[i] for i in range(a, b)]
                for (a, b) in zip(range(0, chars_per_row*num_rows, chars_per_row),
                                  range(chars_per_row, chars_per_row*(num_rows+1), chars_per_row))]
        for row in rows:
            print(row)
        for row in rows:
            row[0] = self.escape_line_end_space(rows[0])
            row[-1] = self.escape_line_end_space(rows[-1])
        rows = [[self.escape_other_spaces(e) for e in row] for row in rows]
        for r in rows:
            print(r)
        return [f"\\texttt{''.join(r)}\\\\" for r in rows]
        final_latex = '\n'.join(
            [f"\\texttt{''.join(r)}\\\\" for r in rows]
        )
        return final_latex

    # ====================
    @staticmethod
    def escape_line_end_space(entry: str):

        if entry == ' ':
            return r"\Verb+{\ }+"
        else:
            return entry

    # ====================
    @staticmethod
    def escape_other_spaces(entry: str):

        if entry == ' ':
            return r"{\ }"
        else:
            return entry

    # ====================
    @staticmethod
    def feature_display_name(feature):
        """Return the display name for a feature."""

        if feature in FEATURE_DISPLAY_NAMES:
            return FEATURE_DISPLAY_NAMES[feature]
        else:
            return f"'{feature}'"
