# ml_engineer


This project creates a docker container Docker container for training and calling a linear regression model based on the housing prices dataset to predict respective house prices.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requeriments.

```bash
pip install -r requeriments.txt
```

**Docker:** http://dockerhub.com

**Github CI** http://github.com) 

## Usage

Use the download data argument  <em>--download-data</em> to load the boston dataset , <em>run-model</em>to train the model, and <em>run-all</em> to launch the entire pipeline.

**Boston dataset:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

```python
from data.load_data import get_data
from model.model import run_model
import sys


def main():
    arg = sys.argv[1]
    if arg == '--download-data':
        get_data()
    if arg == '--run-model':
        run_model()
    if arg == '--run-all':
        get_data()
        run_model()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)