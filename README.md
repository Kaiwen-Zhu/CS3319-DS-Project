# CS3319-DS-Project
Project of CS3319: Foundations of Data Science, Spring 2023

## Setup
```sh
conda create -n ds-project python=3.10
activate ds-project
pip install torch
pip install matplotlib
pip install scikit-learn
pip install dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Train
+ Without edge enhancement:
```sh
python main.py
```
+ With simple edge enhancement:
```sh
python main.py --add_write
```
+ With further edge enhancement:
```sh
python main.py --enhance
```

## Test
```sh
python evaluate.py --test_path {filepath}
```
Here `filepath` is the path of the `txt` file containing edges to be predicted. For example, if the file is placed at the same directory as this README file with name `test.txt`, then just run
```sh
python evaluate.py --test_path test.txt
```
and the result will be written to `./Submission.csv`.

Contact us if there are any problems.