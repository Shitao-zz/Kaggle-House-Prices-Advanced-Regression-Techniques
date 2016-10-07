# Kaggle House Prices: Advanced Regression Techniques
[House Prices: Advanced Regression Techniques
Competition on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
#### Shitao Wang submission version 1

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://stanford.edu/~mwaskom/software/seaborn/#)
- [scikit-learn](http://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

All ipython notebook are used for data preprocessing, feature transforming and outlier detecting.
All core scripts are in `code` folder, in which the ensemble learning script is in `ensemble` folder and
base model script is in `sing_model` folder. All input data are in `input` folder and the detailed description
of the data can be found in [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). 
### Run

For a single model run, navigate to the `/code/single_model/` and run the following commands: ```python base_model.py```
For a ensemble run, navigate to the `/code/ensemble/` and run the following commands: ```python ensemble.py```
Make sure to change the data directory and the parameters accordingly before the model run.

### Submission

<img src="/doc/fig/score_with_different_approaches.png">

Submission score on Kaggle leaderboard with different approaches.

### FlowChart

<img src="/doc/fig/flowchart.jpg" align="center" width="900px"/>

Flow chart of the code.

### Documentation

See `./doc/capstone_doc.pdf` for detailed project documentation.
