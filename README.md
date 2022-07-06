# Precision & recall calculator

A Python class for calculating precision, recall and F-score metrics for the outputs of feature restoration models against reference strings.

## Getting started

1. Clone the repository

Recommended method for Google Colab notebooks:

```python
import sys
# Delete precision-recall-calculator folder to ensures that any changes to the repo are reflected
!rm -rf 'precision-recall-calculator'
# Clone precision-recall-calculator repo
!git clone https://github.com/ljdyer/precision-recall-calculator.git
# Add precision-recall-calculator to PYTHONPATH
sys.path.append('precision-recall-calculator/src')
```

2. Install requirements (if required)

There is no need to install any libraries in Google Colab, as all required libraries are already pre-installed by default.

If working in a virtual environment, run the following in the src directory:

python```
pip install -r requirements.txt
```

3. Import PrecisionRecallCalculator class

python```
from main import PrecisionRecallCalculator
```

## Initializing a class instance

python```
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
        Initialize an instance of PrecisionRecallCalculator class

        Required arguments:
        -------------------
        reference: Str_or_List_or_Series    Either a single string, or a list
                                            or pandas.Series object of strings
                                            ('documents') to use as the
                                            reference corpus.
        hypothesis: Str_or_List_or_Series   Either a single string, or a list
                                            or pandas.Series object of strings
                                            ('documents') to use as the
                                            hypothesis corpus.
                                            (Number of documents must be the
                                            same as reference.)
        capitalisation: bool                Whether or not to treat
                                            capitalisation as a feature to be
                                            assessed.
        feature_chars: Str_or_List          A string or list of characters
                                            containing other characters to
                                            treat as features (e.g. '., ' for
                                            periods, commas, and spaces.)

        Optional keyword arguments:
        ---------------------------
        get_cms_on_init: bool               Whether or not to get confusion
                                            matrics for all reference/
                                            hypothesis documents on
                                            intiialization.
                                            Set to false and access manually if
                                            only looking at metrics for a
                                            subset of documents in a large
                                            corpus.
        """
```
