# Prediction-of-confirmed-cases-using-AdaBoost-Regressor-for-Covid19-dataes-using-

1. **Data Preparation and Exploration:**

   - Import necessary libraries such as NumPy, Seaborn, Matplotlib, and Pandas.
   - Read the COVID-19 data from a CSV file ('covid_19_india.csv') into a Pandas DataFrame (`df`).
   - Display the first few rows of the DataFrame using `df.head()`.
   - Get information about the DataFrame using `df.info()`.
   - Calculate summary statistics of the DataFrame using `df.describe()`.
   - Calculate the 'Active' cases by subtracting 'Cured' and 'Deaths' from 'Confirmed'.
   - List the columns in the DataFrame using `df.columns`.
   - List unique values in the 'State/UnionTerritory' column.
   - Calculate the correlation matrix of the DataFrame using `df.corr()`.
   - Drop the 'Date' column from the DataFrame using `df=df.drop('Date',axis=1)`.
   - Separate object-type columns and numeric-type columns into two DataFrames (`my_object` and `my_numeric`).
   - Perform one-hot encoding on object-type columns and concatenate the resulting DataFrame (`my_obj_con`) with the numeric DataFrame (`my_numeric`) to create a final DataFrame (`df_fin`).

2. **Data Visualization:**

   - Create a barplot using Seaborn to visualize the 'Confirmed' cases by 'State/UnionTerritory'.
   - Customize the plot, including the figure size and x-axis rotation.

3. **Machine Learning - Elastic Net Regression:**

   - Import the `train_test_split` function from Scikit-Learn to split the data into training and testing sets.
   - Define the features (X) and the target variable (y).
   - Initialize an Elastic Net model.
   - Define a parameter grid for hyperparameter tuning.
   - Import `GridSearchCV` for hyperparameter tuning.
   - Initialize a StandardScaler for feature scaling.
   - Scale the training and testing data using the StandardScaler.
   - Fit the Elastic Net model using GridSearchCV on the training data.
   - Make predictions on the testing data.
   - Calculate the Root Mean Squared Error (RMSE) as the evaluation metric.
   - Display the best estimator from the hyperparameter tuning.

4. **Machine Learning - AdaBoost Regression:**

   - Import the `AdaBoostRegressor` from Scikit-Learn.
   - Initialize an AdaBoost Regressor model.
   - Fit the model on the scaled training data.
   - Make predictions on the testing data.
   - Calculate RMSE as the evaluation metric.

5. **Conclusion:**

   - Conclude that the best model among the two is Elastic Net with an L1 ratio of 1 (Lasso Regression).
