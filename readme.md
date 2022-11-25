Machine Learning CW 2 - Neural Networks
===========================
This repository contains a python implementation of the Neural Networks coursework for the "Introduction to Machine Learning" Autumn module. It consists of a neural network mini-library and a neural network trained on the California House Prices Dataset to predict housing prices for a given datapoint.

Contributors
------------
- Shaheen Amin (sa2920)
- Shaanuka Gunaratne (sg1920)
- Indraneel Dulange (ikd120)
- Omar Zeidan (oz20)

Overview
--------
1. Neural Network Mini-library  
The neural network mini-library is located here [**part1_nn_lib.py**](part1_nn_lib.py).

2. Trained Neural Network  
The trained neural network implementation can be found here [**part2_house_value_regression.py**](part2_house_value_regression.py). The command `python3 part2_house_value_regression.py` can be used to run this file. By default, this program will load and train on the [**housing.csv**](housing.csv) file, while also maintaining a validation set for scoring. The model will then be evaluated on the validation set and the RMSE will be outputted.

Note: By default, hyperparameter tuning is disabled; uncommenting lines 405 and 406 will allow it.

[**hyperparamhelper.ipynb**](hyperparamhelper.ipynb)
----------
This notebook was used to assist in the development of the hyperparameter tuner. It walks you through each stage of tuning and allows you to experiment with the data while only producing the models once. This was created to save time and assist in debigging the code and is kept to demonstrate our development techniques.  

[**housing.csv**](housing.csv)
----------
This file provides the raw data used to train the neural network.

[**iris.dat**](iris.dat)
----------
The Iris dataset, which was used to test the neural network mini-library, is included in this file.

[**part2_model.pickle**](part2_model.pickle)
----------
The pickle file contains the neural network that performed the best with the most optimal hyperparameter configuration determined during hyperparameter tuning.
