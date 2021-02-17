# Attentive Reader
Tensorflow implementation of the Attentive Reader proposed in the paper [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858).

### Dependencies

- Python 3.6
- Tensorflow == 1.14.0

### Usage

#### Datasets

- The two processed reading comprehension datasets, CNN and Daily Mail, can be downloaded from [rc-cnn-dailymail](https://github.com/danqi/rc-cnn-dailymail).
- The glove word embedding can be downloaded from [glove.6B](http://nlp.stanford.edu/data/glove.6B.zip).

#### Training

To preprocess the dataset (make sure all the data are in correct directories)

```bash
python data_helpers.py
```

To train the model (please check all the hyper-parameters)

```bash
python train.py
```

To evaluate the trained model

```bash
python eval.py
```


