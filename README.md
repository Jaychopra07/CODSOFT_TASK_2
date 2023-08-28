# README: Iris Flower Classification

## Submitted By: Jay Chopra

## Intern as: Data Science

## Aim

The aim of this project is to develop a model on the basis of given data and characters that can classify iris flowers into different species based on their characteristics and appearance like sepal and petal measurements.

## Libraries Used

The following important libraries were used for this project:

- numpy
- pandas
- sklearn.cluster.KMeans
- matplotlib.pyplot
- seaborn
- sklearn.model_selection.train_test_split
- from sklearn.neighbors.KNeighborsClassifier
- from sklearn.metrics.accuracy_score

## Dataset

The iris dataset was loaded using pandas's read function from a link `'https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv' `, which contains information about iris flowers, including sepal length, sepal width, petal length, petal width, and species.

## Data Exploration and Preprocessing

1. The dataset was loaded using seaborn's `load_dataset` function as a DataFrame, and its first 5 rows were displayed using `df.head()`.
2. Categorising the 'species' column in the DataFrame was encoded to numerical values using `pd.factorize(df['species'])`.
3. Descriptive statistics for the dataset were displayed using `df.describe()`.
4. Missing values in the dataset were checked using `df.isna().sum()`, and we finf thatr there is no missing value in this given dataset.

## Data Visualization

1. 3D scatter plots were created to visualize the relationship between species, petal length, and petal width, as well as between species, sepal length, and sepal width using `matplotlib.pyplot` and `mpl_toolkits.mplot3d.Axes3D`.
2. 2D scatter plots were created to visualize the relationship between species and sepal length, as well as between species and sepal width using `seaborn.scatterplot`.

## Applying Elbow Technique for K-Means Clustering

1. The Elbow Technique was applied to determine the optimal number of clusters (K) using the sum of squared errors (SSE).
2. The KMeans algorithm was initialized with different values of K (1 to 10) and SSE was computed for each K value.
3. A plot of K values against SSE was created using `matplotlib.pyplot` to identify the "elbow point," which indicates the optimal number of clusters.

## Applying K-Means Algorithm

1. The KMeans algorithm was applied to the dataset with the optimal number of clusters (K=3) obtained from the Elbow Technique.
2. The cluster labels were predicted for each data point in the dataset using `km.fit_predict(df[['petal_length','petal_width']])`.

## Accuracy Check

1. The confusion matrix was calculated to evaluate the accuracy of the KMeans clustering.
2. The confusion matrix was plotted using `matplotlib.pyplot.imshow` and `plt.text` to visualize the true and predicted labels.
3. We also used `accuracy_score(y_test, y_pred)` for checking the accuracy after training the model.

## Training and Testing 
1. The `train_test_split` is used to train the model.
2. We divided the dataset given along into two parts the training data and second the testing data.

## Prediction 
1. For predicting the result i.e. for testing we had used `y_pred = classifier.predict(X_test)`.

## Observations
1. We found that the major part of the data is of setosa species
2. W3e also found that the major distinguishing factor is the petal dimensions i.e. petal length and petal width.
3. The accuracy of the model is found to be 100% i.e. the model is completely reliable.
