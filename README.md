# CNN Facies Classifier
*Undergraduate thesis for my Computer Science degree at UFSC*

## Abstract

![cnn result slices](img/cnn_result.gif)

## Setup
After you've created your python virtual environment, run:
```
pip install -r requirements.txt
```

### Stanford VI-E
The raw Stanford VI-E dataset used in this repo is included. To generate the training examples, use the following script (see --help for arguments).
```
python src/data/make_dataset.py
```

### F3-Block
The F3 data used in this repo is avaiable [here](https://github.com/olivesgatech/facies_classification_benchmark). To generate the training examples, use the following script (see --help for arguments).

```
python src/data/make_dataset_f3.py
```

## Running
Jupyter notebooks are located in ```/notebooks```. The trained models are stored in ```/models```.