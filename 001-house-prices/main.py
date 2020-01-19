import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def create_fit_decision_tree_model(train_X, train_y, leaf_nodes):
    # Create a model
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes, random_state=1)
    # Train/Fit the model with the training data
    model.fit(train_X, train_y)
    return model

# Function to find the mean absolute error
def find_mae(train_X, train_y, val_X, val_y, leaf_nodes):   
    model = create_fit_decision_tree_model(train_X, train_y, leaf_nodes) 
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_predictions,val_y)
    print("leaf_nodes = ", leaf_nodes, " mae = ", mae)
    return mae

# Find the best leaf node count based on lowest mae (dictionary comprehension)
def find_best_leaf_nodes(train_X, train_y, val_X, val_y, max_depth):
    return {leaf_node:find_mae(train_X, train_y, val_X, val_y, leaf_node)  for leaf_node in range(2, max_depth) }

# Plot the leaf_node_count vs mae in a graph
def plot_graph(leaf_node_mae_dict):
    list = sorted(leaf_node_mae_dict.items())
    x, y = zip(*list)
    plt.plot(x, y)
    plt.show()

def main():
    TRAINING_DATASET_PATH = "/001-house-prices/input/train.csv"
    TESTING_DATASET_PATH = "/001-house-prices/input/test.csv"
    SUBMISSION_FILE_PATH = "/001-house-prices/output/submission.csv"

    # Read the csv file as panda dataframe
    home_dataframe = pd.read_csv(os.getcwd() + TRAINING_DATASET_PATH)
    print("No of records = ", home_dataframe.shape[0])

    # Print the columns in dataframe
    print(home_dataframe.columns)

    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_dataframe[features]

    # Create y
    y = home_dataframe.SalePrice

    # Split the dataset to training and validation data
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

    # Find the lowest mae and the corresponding leaf node depth
    leaf_node_count_mae_dict = find_best_leaf_nodes(train_X, train_y, val_X, val_y, home_dataframe.shape[0] // 2)

    # Plot the graph for visual experience
    plot_graph(leaf_node_count_mae_dict)

    # Find the leaf_node_count with the lowest mae
    best_leaf_node_count = min(leaf_node_count_mae_dict, key=leaf_node_count_mae_dict.get)
    print("best_leaf_node_count = ", best_leaf_node_count)

    # To improve accuracy, create a new model which trains on all training data
    model_on_full_data = create_fit_decision_tree_model(X,y,best_leaf_node_count)

    # Predict on the actual test data that will be submitted to kaggle
    test_dataframe = pd.read_csv(os.getcwd() + TESTING_DATASET_PATH)
    test_X = test_dataframe[features]
    test_predictions = model_on_full_data.predict(test_X)
    print(test_predictions)

    # Save predictions in format used for kaggle competition scoring
    output = pd.DataFrame({'Id': test_dataframe.Id, 'SalePrice': test_predictions})
    output.to_csv(os.getcwd() + SUBMISSION_FILE_PATH, index=False)

    #Print the submission csv
    print(output.head())
    
if __name__ == "__main__":
    main()