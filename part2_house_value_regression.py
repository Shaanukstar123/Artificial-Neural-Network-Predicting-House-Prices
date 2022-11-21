import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection #For one-hot encoding and GridSearch for hyperparam tuning
import torch.nn as nn

class Regressor():

    def __init__(self, x, nb_epoch=1000, learningRate=0.01, neuronArchitecture=[13,13,13], batchSize=32, minImprovement=0.1, paramDict=None):
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

        self.bin_labels  = preprocessing.LabelBinarizer()
        self.bin_labels.classes = ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","NEAR OCEAN"]
        #nb_epoch = 1000, learningRate=0.01, neuroncount=8, neuronLayers=3, batchSize=512, 
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        #self.input_layer = nn.Linear(neuronArchitecture[0],neuronArchitecture[1])
        self.output_layer = nn.Linear(neuronArchitecture[-1],1)
        self.all_layers = nn.ModuleList()
        self.activation = nn.ReLU()
        #self.all_layers.append(self.input_layer)
        #self.all_layers.append(nn.ReLU())

        for i in range(len(neuronArchitecture)-1): #list of input and all hidden layers
            self.all_layers.append(nn.Linear(neuronArchitecture[i],neuronArchitecture[i+1]))
            self.all_layers.append(self.activation)

        self.all_layers.append(self.output_layer)
        self.model = nn.Sequential(*self.all_layers)
        self.model.double()

        if paramDict:
            self.nb_epoch = paramDict["nb_epoch"]
            self.learningRate = paramDict["learningRate"]
            self.neuronArchitecture = paramDict["neuronArchitecture"]
            self.batchSize = paramDict["batchSize"]
            self.minImprovement = paramDict["minImprovement"]
        else:
            self.nb_epoch = nb_epoch
            self.learningRate = learningRate
            self.neuronArchitecture = neuronArchitecture
            self.batchSize = batchSize
            self.minImprovement = minImprovement
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

        
        proximity_column  = pd.DataFrame(self.bin_labels.fit_transform(x["ocean_proximity"]))
        x = x.drop(columns="ocean_proximity",axis = 1)
        x = x.join(proximity_column)
        if training:
            x=(x-x.min())/(x.max()-x.min()) #Normalises numerical data from a scale of 0-1
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  #allows all rows to be printed

        #converts x and y to tensors before returning
        return torch.from_numpy(x.values), (torch.from_numpy(y.values) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, xValidation=None, yValidation=None, minImprovement=0.01):
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
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        #Mini-batch gradient descent:
        torch.set_printoptions(profile="full")

        for epoch in range(1):#self.nb_epoch):
            batch_list = torch.randperm(len(X)) # generates random indices

            for i in range(0,len(X),self.batchSize):
                print("BATCH: ",X[0])
                optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
               
                index = batch_list[i:i+ self.batchSize]
                batch_x = X[index]
                batch_y = Y[index]
                prediction = self.predict(batch_x)
                batch_loss = nn.MSELoss(prediction,batch_y)
                batch_loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            

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
        return self.model(x)
        # for layer in self.all_layers:
        #     prediction = layer(x)
        #     prediction = nn.functional.relu(prediction)
        # prediction = self.output_layer(x)
        # return prediction

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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



def RegressorHyperParameterSearch(x, y, hyperparam, minImprovement=0.1, candidateThreshold=0.5, iterations=3): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.
    The approach is to start with very wide hyperparameters, and iteratively modify the hyperparamters for the next iteration
    based on the top 'candidateThreshold' % of models in the current iteration. 
    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # iteration = 1
    # while iteration < iterations:
    #     xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(x, y, test_size=0.1, shuffle=True)
    #     iteration += 1
    #     model = model_selection.GridSearchCV(
    #         estimator = Regressor(x),
    #         param_grid = hyperparam,
    #         scoring="neg_root_mean_squared_error",
    #         cv=5,
    #         verbose=2,
    #         n_jobs=-1
    #     )
    #     model.fit(xTrain, yTrain, xValidation, yValidation, minImprovement)

    return  # Return the chosen hyper parameters

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
    #Hyperparameter tuning
    baseparam = {
        "nb_epoch" : 1000, 
        "learningRate" : [0.001, 0.01, 0.1, 1], 
        "neuronArchitecture" : [[12], [12,12], [12,12,12], [12,12,12,12]], 
        "batchSize" : [64, 128, 256, 512],
        }
    RegressorHyperParameterSearch(x_train, y_train, baseparam, 0.1, 0.5, 3)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
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