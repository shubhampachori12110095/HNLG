## Reference
Main papers to be cited:

```
@inproceedings{su2018natural,
    title={Natural Language Generation by Hierarchical Decoding with Linguistic Patterns},
    author={Shang-Yu Su, Kai-Ling Lo, Yi-Ting Yeh, and Yun-Nung Chen},
    booktitle={Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year={2018}
}
```

## Setup

```
$ mkdir data
# take conda for example, create a new environment
$ conda create -n [your_env_name] python=3
$ source activate [your_env_name]
$ conda install pytorch torchvision cuda80 -c soumith
$ conda install spacy nltk
# download the spaCy models
$ python -m spacy download en
```

## Usage:

```
# under src/
$ python3 train.py --data_dir=/path/to/data
```

TensorBoard Support:

```
$ python3 train.py --data_dir=/path/to/data --log_dir=/path/to/log_dir --exp_name=exp_name
$ tensorbaord --logdir=/path/to/log_dir/exp_name
# Open your brower and connect to the url displayed by tensorbaord
```


Optional Arguments:

```
# under src/
$ python3 train.py --help
```

## Data:

### E2E NLG:
[Link](http://www.macs.hw.ac.uk/InteractionLab/E2E/)


