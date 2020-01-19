# Prem' experiments with ML

This repository has the code for all my ML experiements and Kaggle competitions.

## Pre-requisites

* Install [Python3](https://www.python.org)
* Install [Poetry](https://python-poetry.org)

## Instructions

1. Clone the repo to your local machine 

```
git clone git@github.com:smileprem/machine-learning.git
```
2. Create Python virtual environment

```
poetry shell
```
This command will create the virtual env and activate it as well. 

**PS:** Sometimes, I noticed that `poetry shell` command does not activate the venv. In that case, use the below command. 

```
. `poetry env info -p`/bin/activate
```

3. Install the python dependencies

```
poetry install
```

4. Run the ML experiments. Each program resides in its own folder. The path to the input datasets are relative to subfolder. So, always move to the specific folder and run the program.

```
cd 001-house-prices 
python3 main.py
```

5. Deactivate the virtual environment.
```
deactivate
```

## Feedback

Please create a PR/Issue for any feedback/improvements on my programs.
