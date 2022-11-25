Machine Learning CW 2 - Neural Networks
===========================
This repository contains the python implementation of a neural network mini-library and a neural network trained on the California House Prices Dataset to estimate housing prices for a given datapoint. This coursework is part of COMP70050 - Introduction to Machine Learning (Autumn 2022).

Contributors
------------
- Shaheen Amin (sa2920)
- Shaanuka Gunaratne (sg1920)
- Indraneel Dulange (ikd120)
- Omar Zeidan (oz20)

Overview
--------
1. Neural Network Mini-library  
The neural network mini-library can be found here [**part1_nn_lib.py**](part1_nn_lib.py).

2. Trained Neural Network  
The implementation of the trained neural network can be found here [**part2_house_value_regression.py**](part2_house_value_regression.py). This file can be run using the `python3 part2_house_value_regression.py` command. By default, this program will load the [**housing.csv**](housing.csv) file and train on it, whilst also retaining a validation set for scoring after. The model will then be evaluated on the validation set and will output the RMSE. Additionally, hyperparameter tuning is disabled by default - uncommenting lines 405 and 406 will enable this function. 

[**hyperparamhelper.ipynb**](hyperparamhelper.ipynb)
----------
This notebook was used to help create the hyperparameter tuner. It simply goes through each stage of the tuning and allows you to experiment with the data whilst only generating the models once - it was used to save time and debug the code and is retained only to show our development process.

[**iris.dat**](iris.dat)
----------
This file contains the Iris dataset which was used to test the neural network mini-library.

[**housing.csv**](housing.csv)
----------
This file contains the raw data the neural network was trained on.

[**part2_model.pickle**](part2_model.pickle)
----------
The pickle file is the neural network that performed the best with the most optimal comfiguration found for the hyperparameters during hyperparameter tuning.