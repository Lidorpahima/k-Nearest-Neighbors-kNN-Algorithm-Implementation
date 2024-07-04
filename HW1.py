import pandas as pd
import numpy as np
# In this class, we will implement the kNN algorithm and predict the class labels of the test set.
def Normalize(x, min_val, max_val):  # Normalization Formula: (X - Xmin) / (Xmax - Xmin)
    return (x - min_val) / (max_val - min_val)


def EuclideanDistance(x1, x2):  # Euclidean Distance Formula: sqrt((x1 - x2)^2)
    return np.sqrt(np.sum((x1 - x2) ** 2))

def kNN_predict(train_data,test_data,k):
    #Load the data from the csv files
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    #we will need seprate features and class label
    train_features = train_df.iloc[:,:-1]
    train_class = train_df.iloc[:,-1]
    #test features dont have class label so we will take all the columns
    test_features = test_df.iloc[:,:]
    #we will find the max and min values of the features in the training set to normalize the data
    min_val = train_features.min()
    max_val = train_features.max()
    #Normalize the data using the min and max values of the training set
    train_features = Normalize(train_features,min_val,max_val)
    test_features = Normalize(test_features,min_val,max_val)
    #initialize the list of predictions
    prob = [] 
    for test_sample in test_features.values:
        #calculate the distance between the test sample and all the training samples
        distances = []
        for train_sample in train_features.values:
            distances.append(EuclideanDistance(test_sample,train_sample))#adding to array distance between test sample and train sample
        #sort the distances and take the k nearest neighbors
        neighbors = np.argsort(distances)[:k]
        #find the class label of the k nearest neighbors from what we have in the training set
        neighbors_class = train_class.iloc[neighbors]
        #find the most common class label in the k nearest neighbors
        prediction = np.mean(neighbors_class)
        prob.append(prediction)
    return prob
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def main():
    train_path = "train_set.csv" # Path to the training set
    test_path = "test_set.csv" # Path to the test set
    k = 5 # Number of neighbors (K)
    predictions = kNN_predict(train_path,test_path,k) # Predictions will store the predicted class labels using kNN algorithm
    predictions = [float(prediction) for prediction in predictions] # Convert the predictions to float because it will return as numpy array
    print(predictions) # Print the predictions
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main call ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#    
if __name__ == "__main__": 
    main()
