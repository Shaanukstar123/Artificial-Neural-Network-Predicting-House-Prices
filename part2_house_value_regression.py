import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection #For one-hot encoding and GridSearch for hyperparam tuning
import collections
import math
import random
import torch.nn as nn

class Regressor():
    def __init__(self, x=None, nb_epoch=100, learningRate=0.01, neuronArchitecture=[13,9], batchSize=64, minImprovement=0.00005, paramDict=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size P
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #Hyperparameter setting
        self.minImprovement = minImprovement
        if paramDict is None:
            #Default values
            paramDict = {
            "nb_epoch" : nb_epoch, 
            "learningRate" : learningRate,
            "neuronArchitecture" : neuronArchitecture, 
            "batchSize" : batchSize
            }
        self.paramDict = paramDict
        self.set_params(**self.paramDict)
        #Convert string labels to numerical
        self.testing_labels = None
        #Early stop parameters
        self.allowance = 5
        self.count = 0
        self.prevLoss = math.inf
        #epoch plotter
        self.epochData = [[], [], []]
        return
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    def _preprocessor(self, x, y = None, training = False):

        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        #Fills empty data points with averages of their column
        pd.options.mode.chained_assignment = None
        for col in x:
            if col in {"longitude","latitude", "median_income"}:
                x[col].fillna(x[col].mean(), inplace=True)
            elif col == "ocean_proximity":
                x[col].fillna(x[col].mode()[0], inplace=True)
            else:
                x[col].fillna(x[col].median(), inplace=True)

        #binarises textual elements
        if training:
            training_labels = preprocessing.LabelBinarizer()
            training_labels.classes_ = ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","NEAR OCEAN"]
            proximity_column  = pd.DataFrame(training_labels.fit_transform(x["ocean_proximity"]))
            self.testing_labels = training_labels
        else:
            #uses saved binarizer from training in case testing data doesn't contain all ocean proximity classes
            proximity_column  = pd.DataFrame(self.testing_labels.transform(x["ocean_proximity"])) 


        #print("proximity_col: ",proximity_column)
        x.reset_index(drop=True, inplace=True)
        x = x.drop(columns="ocean_proximity",axis = 0)
        x = x.join(proximity_column)
        #print("Postprocessed")
        #print(x)
        if training:
            #Determine scaling factors
            self.xMin = x.min()
            self.xMax = x.max()
            self.xRange = self.xMax-self.xMin
        #Normalises numerical data from a scale of 0-1
        x = (x-self.xMin)/(self.xRange)
        #converts x and y to tensors before returning
        return torch.from_numpy(x.values), (torch.from_numpy(y.values) if isinstance(y, pd.DataFrame) else None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    def fit(self, x, y, xValidation=None, yValidation=None, plotData=False):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #Create network
        #Ensure first layer contains 13 neurons to match input feature size
        self.neuronArchitecture = [13] + self.neuronArchitecture
        #Neuron architecture
        self.output_layer = nn.Linear(in_features=self.neuronArchitecture[-1],out_features=1)
        self.layer_list = []
        for i in range(len(self.neuronArchitecture)-1): #list of input and all hidden layers
            self.layer_list.append(nn.Linear(in_features=self.neuronArchitecture[i],out_features=self.neuronArchitecture[i+1]))
            self.layer_list.append(nn.ReLU())
        self.layer_list.append(self.output_layer)
        self.model = nn.Sequential(*self.layer_list) #unpacks list as parameters for sequential layers
        self.model.apply(self.init_weights)
        self.model.to(torch.float64)
        network = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        lossFunc = nn.MSELoss()
        #Preprocess training data to generate scalars
        X, Y = self._preprocessor(x, y = y, training = True)
        #Mini-batch gradient descent:
        torch.set_printoptions(profile="full")
        currEpoc = 0
        while currEpoc < self.nb_epoch:
            batch_list = torch.randperm(len(X)) # generates random indices
            print(currEpoc, end='-')
            for i in range(0,len(X),self.batchSize):
                network.zero_grad()
                index = batch_list[i:i+self.batchSize]
                batch_x = X[index]
                batch_y = Y[index]
                prediction = self.model(batch_x)
                batch_loss = lossFunc(prediction,batch_y) # MSELoss is a wrapper function
                batch_loss.backward()
                network.step()
            currEpoc += 1
            #Use the validation set to implement early stopping - used during hyperparamter tuning
            if xValidation is not None:
                newError = self.score(xValidation, yValidation)
                if self.earlyStop(newError) and not plotData:
                    print("Reached epoch cycle:", currEpoc, "with error:", newError)
                    break
                if currEpoc > 1 and plotData:
                    trainError = self.score(x, y)
                    self.epochData[0].append(currEpoc)
                    self.epochData[1].append(newError)
                    self.epochData[2].append(trainError)
        return
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        
        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x)
        output = self.model(X).detach().numpy()
        return output
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y)
        yPred = self.model(X)
        diff = yPred-Y
        #Calculate RMSE
        total = 0
        for element in diff:
            total += element**2
        return math.sqrt(total/len(diff))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    # All helper functions in class
    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
    
    def earlyStop(self, validationLoss):
        if validationLoss*(self.minImprovement+1) < self.prevLoss:
            self.prevLoss = validationLoss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.allowance:
                return True
        return False

    def get_params(self, deep=False):
        return self.paramDict
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

# Helper functions
#This function will find the top two paramters and create a range between them
#This function will find the top two paramters and create a range between them
def getTopTwo(inputList):
    paramHeaders = {"nb_epoch" : 0, "learningRate" : 1, "neuronArchitecture" : 2, "batchSize" : 3}
    params = [[] for i in range(len(paramHeaders))]
    paramMode = dict()
    #Convert list of dictionaries to list per parameter
    for description in inputList:
        for key, value in description.items():
            params[paramHeaders[key]].append(tuple(value) if isinstance(value, list) else value)
    #Invert the dictionary
    paramInverted = {value: key for key, value in paramHeaders.items()}
    #Obtain the two most common items
    for index, value in enumerate(params):
        mode = collections.Counter(params[index]).most_common(2)
        paramMode[paramInverted[index]] = [i[0] for i in mode]
    return paramMode

def RegressorHyperParameterSearch(x, y, hyperparam, candidateThreshold=0.05, iterations=2, wideSearch = True): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.
    The approach is to start with very wide hyperparameters, and iteratively modify the hyperparamters for the next iteration
    based on the top 'candidateThreshold' % of models in the current iteration.
    The primary goal of the first iteration is to determine how many layers of neurons should be used and the order of magntiude for the learning rate
    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    iteration = 0
    bestPerformer = -math.inf
    bestParams = hyperparam
    while iteration < iterations:
        xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(x, y, test_size=0.1)
        iteration += 1
        model = model_selection.GridSearchCV(
            estimator = Regressor(),
            param_grid = hyperparam,
            scoring="neg_root_mean_squared_error", #Scoring metric means lower is better
            cv=5,
            verbose=4,
            n_jobs=5,
            error_score='raise',
            return_train_score = True
            )
        model.fit(xTrain, yTrain, xValidation=xValidation, yValidation=yValidation)
        results = pd.DataFrame(model.cv_results_) #Get results
        currentPerformer = results["mean_test_score"].max() #Find best performer from models
        #If the newest iteration has a worse performance, terminate tuning and return the last one
        print("Best performer:", currentPerformer)
        if abs(currentPerformer) > abs(bestPerformer):
            return bestParams
        bestPerformer = currentPerformer
        bestParams = model.best_params_
        if (iteration == iterations):
            return bestParams
        #Get all models within 'candidateThreshold' % of best performance
        resultsTop = results.loc[results["mean_test_score"] >= bestPerformer*(1+candidateThreshold)]
        paramList = resultsTop["params"].values
        #Now, calculate all the new hyperparameters and prepare for next round
        newParams = getTopTwo(paramList)
        #On the first iteration, determine magnitude of learning rate and the amount of layers in the neural network
        if iteration == 1:
            #Find magnitudes
            #Determine the amount of layers - prefer less layers
            layerCount = min([len(x) for x in newParams["neuronArchitecture"]])
            #Determine the magnitude of the learning rate 
            if len(newParams["learningRate"]) >= 2:
                learningMagnitude = sum([math.log(x, 10) for x in newParams["learningRate"][:2]])/2
            elif len(newParams["learningRate"]) == 1:
                learningMagnitude = math.log(newParams["learningRate"][0], 10)
            else:
                return bestParams
        print("Layercount:", layerCount, "Learning Magnitude:", learningMagnitude, "Learning rate approx:", 10**learningMagnitude)
        hyperparam = {"nb_epoch" : None, "learningRate" : None, "neuronArchitecture" : [], "batchSize" : None}
        magnitudeModifier = 0.6
        neuronModifier = 3
        hyperparam["learningRate"] = [10**random.uniform(learningMagnitude-magnitudeModifier, learningMagnitude+magnitudeModifier) for _ in range(4)]
        #Neuron architecture
        for i in range(4):
            maxNeurons = 13
            architecture = []
            for j in range(layerCount):
                #Ensure decreasing neurons
                maxNeurons = random.randint(maxNeurons-neuronModifier, maxNeurons)
                architecture.append(maxNeurons)
            hyperparam["neuronArchitecture"].append(architecture)
        #Batchsize
        if len(newParams["batchSize"]) >= 2:
            batchMagnitude = sum([math.log(x, 2) for x in newParams["batchSize"][:2]])/2
        else:
            batchMagnitude = math.log(newParams["batchSize"][0], 2)
        hyperparam["batchSize"] = [int(2**random.uniform(batchMagnitude-magnitudeModifier, batchMagnitude+magnitudeModifier)) for _ in range(4)]
        hyperparam["nb_epoch"] = newParams["nb_epoch"]
        print("New hyperparameters:", hyperparam)
    hyperparam["nb_epoch"] = hyperparam["nb_epoch"][0]
    return bestParams # Return the chosen hyper parameters
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html




def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(x_train, y_train, test_size=0.1)
    sample = x_train.iloc[0:2]
    #print(sample)
    #Hyperparameter tuning
    hyperparam = {
        "nb_epoch" : [100], 
        "learningRate" : [0.001, 0.01, 0.1], 
        "neuronArchitecture" : [[9], [9,9], [9,9,9]], 
        "batchSize" : [64, 128, 256, 512],
        }
    hyperparam = RegressorHyperParameterSearch(xTrain, yTrain, hyperparam, candidateThreshold=0.05, iterations=2)
    print("Optimum parameters:", hyperparam)
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(xTrain, paramDict=hyperparam)
    regressor.fit(xTrain, yTrain, xValidation, yValidation)
    print(regressor.get_params())
    #print()
    #print(regressor.predict(sample))
    #regressor.score(x, y) #need this to compare against parameter tuning maybe make held out dataset?
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


        

if __name__ == "__main__":
    example_main()

## Sources: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
##          https://pandas.pydata.org/
##          https://pytorch.org/docs/
##          https://www.projectpro.io/recipes/optimize-function-adam-pytorch
##          https://stackoverflow.com/questions/32896651/pass-multiple-arguments-in-form-of-tuple
##          https://rubikscode.net/2021/08/02/pytorch-for-beginners-building-neural-networks/
##          https://scikit-learn.org/stable/developers/develop.html