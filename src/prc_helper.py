"""
helper.py

Helper functions for PrecisionRecallCalculator class
"""

from typing import Any, Union

import pandas as pd
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import numpy as np

Int_or_Str = Union[int, str]
Str_or_List = Union[str, list]
Str_or_List_or_Series = Union[str, list, pd.Series]

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
FEATURE_DISPLAY_NAMES_LATEX = {
    'CAPITALISATION': "CAPS",
    ' ': r"Spaces ('{\ }')",
    ',': "Commas (',')",
    '.': "Periods ('.')",
    'all': 'All'
}


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
def get_tqdm() -> type:
    """Return tqdm.notebook.tqdm if code is being run from a notebook,
    or tqdm.tqdm otherwise"""

    if is_running_from_ipython():
        tqdm_ = notebook_tqdm
    else:
        tqdm_ = non_notebook_tqdm
    return tqdm_


# ====================
def is_running_from_ipython():
    """Determine whether or not the current script is being run from
    a notebook"""

    try:
        # Notebooks have IPython module installed
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False


# ====================
def display_or_print(obj: Any):
    """'print' or 'display' an object, depending on whether the current
    script is running from a notebook or not."""

    if is_running_from_ipython():
        display(obj)
    else:
        print(obj)


# ====================
def label_fps_and_fns(strings: str, features: list, feature_chars: list):

    output_chars = []
    while strings['ref'] and strings['hyp']:
        features_present = {'ref': [], 'hyp': []}
        next_char = {
            'ref': strings['ref'].pop(0), 'hyp': strings['hyp'].pop(0)}
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
            if ('CAPITALISATION' in features
               and next_char[string].isupper()):
                features_present[string].append('CAPITALISATION')
            while (len(strings[string]) > 0
                    and strings[string][0] in features):
                features_present[string].append(strings[string].pop(0))
        if 'CAPITALISATION' in features:
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
        for feature in feature_chars:
            if (feature in features_present['ref']
               and feature not in features_present['hyp']):
                output_chars.append(f'\\fn{{\\mbox{{{feature}}}}}')
            elif (feature not in features_present['ref']
                    and feature in features_present['hyp']):
                output_chars.append(f'\\fp{{\\mbox{{{feature}}}}}')
            elif (feature in features_present['ref']
                    and feature in features_present['hyp']):
                output_chars.append(feature)
    return output_chars


# ====================
def precision_recall_fscore_from_cm(cm: np.ndarray):
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
