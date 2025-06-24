Our data source is solar panel production from major solar facilities in Canada. We want to understand how to classify outliers within this solar production data. The data is collected hourly across several years from 2015 through April 2023. For our analysis, we will reduce the scope to data starting in 2021 and for only the largest facility. A time-series view of this facility's solar kWh production is shown below.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/bearspaw_ts.png" width=100% height=100%>

First, let's use reinforcement learning, namely Q-learning from OpenAI Gym, to diagnose time series anomalies. Using conservative values for the Hyperparameters to reduce the number of diagnosed outliers yields a contamination (or, outlier) rate of 0.55% (53 out of 9,716 observations). 

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/anomaly_ts_q_learning.png" width=100% height=100%>

Next, we can compare the Q-Learning results to the results from an Isolation Forest model. We can define several features for this model considering a window of 12 hours. These metrics are shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/isolation_forest_features.png" width=25% height=25%>

Next, we can estimate the contamination by using 3 standard deviations from the mean for each feature, then averaging the number of outliers over total oberservations across all metrics. The estimated contamination rate is very similar at 0.51% (50 out of 9,716 observations)!

After training the Isolation Forest model using this estimated outlier rate, the result is shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/anomaly_ts_isolation_forest.png" width=100% height=100%>

When comparing models, the outlier rate, as mentioned, is very similar. However, none of the outliers overlap between models. This warrants 

* EDA to understand the types of outliers that are being flagged for each model
* Correctly scoring the Isolation Model on the Evaluation Set using the rule: outliers = np.sum(np.abs(feature_values - mean) > 3 * std)
* Further tuning to both models to attempt some level of convergence of outliers identified

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/model_anomaly_summary1.png" width=50% height=50%>
