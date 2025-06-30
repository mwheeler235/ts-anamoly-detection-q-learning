## Data Preparation
Our data source is solar panel production from major solar facilities in Canada. We want to understand how to classify outliers within this solar production data. The data is collected hourly across several years from 2015 through April 2023. For our analysis, we will reduce the scope to data starting in 2021 and for one plant. Also since nightly hours do not have solar production, we will aggregate the data by date. A time-series view of this plant's solar kWh production is shown below.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/whitehorn_ts.png" width=100% height=100%>

## Reinforcement Learning for Outlier Detection
First, let's use reinforcement learning, namely Q-learning from OpenAI Gym, to diagnose time series anomalies. 

Q-learning can be used for anomaly detection by training an agent to learn a policy that maximizes rewards for normal behavior and minimizes rewards for anomalous behavior. The agent learns to distinguish between normal and abnormal patterns through trial and error, receiving rewards or penalties based on its actions. Anomalies are identified as deviations from the learned normal behavior.

We can use conservative values for the Hyperparameters to reduce the number of diagnosed outliers, thus the model yields a contamination (or, outlier) rate near 2%. 

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/ql_anomaly.png" width=100% height=100%>

## Isolation Forest for Outlier Detection
Next, we can compare the Q-Learning results to the results from an Isolation Forest model. 

Isolation forests are an unsupervised ML approach that we can use for anomaly detection. Using a random subset of the data, the model builds multiple decision trees. Subsampling is used to select a random subset of the data to build each tree, allowing for efficiency and increased regularization. After selecting a random feature from the data, the model chooses a random split point creating two branches. This process is recursively repeated for each new branch until a maximum depth is reached. After the trees are generated, for each tree, the algorithm calculates the path length of each data point to the root of the tree. The average of all trees for each point defines its "anomaly score." Finally, observations with a lower anomaly score are more “normal” and are less likely to be considered anomalies.

Primarily based on the kWh feature, a feature set was generated for this model. For the rolling mean, the window chosen is 7 days in order to capture a weekly "expectation." These metrics are shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/isolation_forest_features1.png" width=25% height=25%>

Next, we can estimate the contamination rate by using 1.75 standard deviations from the mean for each feature, then averaging the number of outliers over total oberservations across all metrics. The estimated contamination rate is very similar at 2.6%. However, for the Isolation Forest model, we will use the exact outlier rate derived from the Q-Learning result (2.17%).

After training the Isolation Forest model, the results are shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/if_anomaly.png" width=100% height=100%>

## Rolling Mean for Outlier Detection

Using the rolling mean is the most logical metric we should consider for a more simplified model. For this analysis, we can define the window to be the previous 7 days. Then in order to yield an outlier ratio close to the results for the other two models, we can set the outlier threshold to be 1.75*(standard deviation). That is, if an observed value differs from the past 7-day mean by more than 1.75 times the standard deviation (of the rolling mean), then we flag the value as an outlier.

## Comparing All 3 Models

When comparing the three models, the outlier rates are quite similar. Below is a chart showing all three outlier models compared:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/ts_model_comp1.png" width=100% height=100%>

To get a better sense of how the models compare, let's zoom in to an anomolous period in 2021.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/ts_model_comp_zoom.png" width=100% height=100%>

Now it's clear that the Isolation Forest is capturing the most reasonable outliers. Additionally, about half of the Isolation Forest outliers are also captured by the Rolling Mean model. The Q-Learning model is clearly not capturing "true" outliers in this set, thus further tuning is certainly warranted. Below is table showing the percent of outliers in each model bucket.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/model_anomaly_summary2.png" width=90% height=90%>

## Observations

* For the Isolation Forest's outliers, about half of them are picked up by the Rolling Mean model. Only one from the Isolation Forest overlapped with the Q-Learning outlier bucket.
* The majority of the Q-Learning outliers were not flagged by the other two models, indicating that further tuning of the RL model is required to approach more congruence between models.

## Time Series Modeling

As a bonus, let's see how predictable daily kWh production is for this solar plant. For the training data, we will choose all of the data from 2015 through 2020, then we can predict the period for 2021 through March of 2023. We also will perform a hyperparameter tuning job for changepoint_prior_scale and seasonality_prior_scale values to optimize the model performance.  Below are the prediction results.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/prophet_preds1.png" width=100% height=100%>

Clearly, the stochastic nature of daily output is difficult for a model to pick up, however, the trend looks very appropriate. The Mean Absoluate Error (MAE) for the test set around 470 kWh and the Weighted Mean Absolute Percentage Error is 41.87%. If we were to model at the monthly level, the MAE and MAPE values would be much lower.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/prophet_eval_metrics.png" width=50% height=50%>


## Anomaly Detection Conclusion

This is an interesting model comparison using daily solar production data. Based on these results, the Isolation Forest is the true winner as it yields:
* Interpretability, given the user can choose the feature set
* Flexibiliity using hyperparameters to "control" the outlier rate and how "deep" the trees go
* Ability to define training and test sets and thus, perform cross-validation to reduce bias

Note that looking at the raw hourly data for outliers would yield more actionable insights as this would allow an organization to make more near-real-time decisions. Next steps should include:
* Modeling at the hourly level with considerations on how to handle (or drop) nightly hours and smooth "bad" training data
* Modeling for other plants to understand how each model performs at different locations
* Cross-validation to determine optimal expected outlier rates
* Assessing further interpretability for the Isolation Forest and Q-Learning models