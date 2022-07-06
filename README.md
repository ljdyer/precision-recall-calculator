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
!git clone https://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@github.com/ljdyer/precision-recall-calculator.git
# Add precision-recall-calculator to PYTHONPATH
sys.path.append('precision-recall-calculator/src')
```

2. Install requirements (if required)

All required libraries are pre-installed by default in Google Colab, so there is no need 

If working in a virtual environment, running the following in the src directory:

```python
pip install -r requirements.txt
```

3. Import PrecisionRecallCalculator class

python```
from main import PrecisionRecallCalculator
```

## Initializing a class instance



