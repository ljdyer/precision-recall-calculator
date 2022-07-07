"""
test.py

Basic tests for PrecisionRecallCalculator class
"""

from precision_recall_calculator_main import PrecisionRecallCalculator

# ====================
if __name__ == "__main__":

    reference = [
        'This is a sentence.',
        'This is another sentence.',
        'This is Sentence 3'
    ]
    hypothesis = [
        'This is a sentence...',
        'This IS another sentence.',
        'Thisis Senten ce 3'
    ]
    prc = PrecisionRecallCalculator(
        reference, hypothesis, capitalisation=True, feature_chars='., '
    )

    sent_0_commas = \
        prc.precision_recall_fscore_from_cm(prc.confusion_matrices[0][','])
    # No commas in reference sentence, so all scores should be N/A
    assert sent_0_commas['Precision'] == 'N/A'
    assert sent_0_commas['Recall'] == 'N/A'
    assert sent_0_commas['F-score'] == 'N/A'

    sent_0_periods = \
        prc.precision_recall_fscore_from_cm(prc.confusion_matrices[0]['.'])
    # Extra periods should be ignored, so precision is 1.
    assert sent_0_periods['Precision'] == 1
    assert sent_0_periods['Recall'] == 1
    assert sent_0_periods['F-score'] == 1

    sent_1_capitalisation = prc.precision_recall_fscore_from_cm(
        prc.confusion_matrices[1]['CAPITALISATION'])
    # 1 true positive, 2 false positives, 0 false negatives
    # Precision is tp/(tp+fp) = 1/(2+1) = 1/3
    # Recall is tp/(tp+fn) = 1/(1+0) = 1
    # F-score is (2*p*r)/(p+r) = (2/3)/(4/3) = 1/2
    assert round(sent_1_capitalisation['Precision'], 2) == 0.33
    assert sent_1_capitalisation['Recall'] == 1
    assert sent_1_capitalisation['F-score'] == 0.5

    sent_2_spaces = \
        prc.precision_recall_fscore_from_cm(prc.confusion_matrices[2][' '])
    # 2 true positives, 1 false positive, 1 false negative
    # Precision is tp/(tp+fp) = 2/(2+1) = 2/3
    # Recall is tp/(tp+fn) = 2/(2+1) = 2/3
    # F-score is (2*p*r)/(p+r) = (4/9)/(2/3) = 2/3
    print(sent_2_spaces)
    assert round(sent_2_spaces['Precision'], 2) == 0.67
    assert round(sent_2_spaces['Recall'], 2) == 0.67
    assert round(sent_2_spaces['F-score'], 2) == 0.67

    print("All tests passed.")
