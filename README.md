# AIR-QUALITY-INDEX
PREDICTION | HOPSWORKS | XGBOOST | WEBAPP

Approach:

The API has city as key. It can also be accesses using idx. SO, I created a list of idx.

There is no key like 'date' or 'day' using which I could fetch data for a particular day. So, I fetched data for all the cities for a particular day and then filtered out the data for a particular city.

Fetching data for all the cities by idx(around 959) using a for loop would take a lot of time. So, I used pyspark to parallelize the process.

Later during feature extraction, I realized that the the dat is inconsistent. Data for some attributes like location, air quality index and the parameters used to calculate air quality index are missing. So, I used the logic to join the data after fethching all the features on location.

Creating a dataframe from the joined spark dataframe was taking a lot of time. So, I decided to complete the task using 100 idx in order to complete the task in time and at least be able to submit the assignment.

Follwed the Hopsworks documentation to train the model and predict on the validation dataset.
