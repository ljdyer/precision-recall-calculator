from jiwer.measures import _get_operation_counts, _preprocess
from jiwer.transformations import wer_default
from misc_helper import display_or_print
import pandas as pd


# ====================
def wer_info(ref: str, hyp: str) -> dict:
    """Calculate reference length, minimum number of edits, and word
    error rate for a single reference+hypothesis pair."""

    len_ref = len(ref.split())
    num_edits = get_num_edits(ref, hyp)
    wer_ = wer(num_edits, len_ref)
    return {
        'len_ref': len_ref,
        'num_edits': num_edits,
        'wer': wer_
    }


# ====================
def get_num_edits(ref: str, hyp: str) -> dict:
    """Get the minimum numbers of word edits required to get from
    hypothesis to reference string."""

    # jiwer library _preprocess with wer_default strips leading and
    # trailing whitespace, splits on space characters, maps words
    # to unique characters and joins together as string so that
    # python-Levenshtein library can be used to calculate WER
    ref_, hyp_ = _preprocess(ref, hyp, wer_default, wer_default)

    # _get_operation_counts returns hits, deletions, substitions,
    # and insertions
    _, S, D, I = _get_operation_counts(ref_[0], hyp_[0])
    edits = sum([S, D, I])

    return edits


# ====================
def wer(num_edits: int, len_ref: int) -> float:
    """Calculate WER for minimum number of edits and reference length"""

    return num_edits / len_ref * 100


# ====================
def show_wer_info_table(wer_info: dict):
    """Show WER info in a table"""

    row_labels = [
        'Length of reference (words)',
        'Minimum edit distance (S+D+I)',
        'Word error rate (%)'
    ]
    wer_info_ = [
        f"{wer_info['len_ref']:,}",
        f"{wer_info['num_edits']:,}",
        f"{(wer_info['err']*100):.3f}%"
    ]
    display_or_print(pd.DataFrame(
        wer_info_, index=row_labels, columns=['Value']))
