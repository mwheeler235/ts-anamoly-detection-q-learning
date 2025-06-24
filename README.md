Our data source is solar panel production from a major solar facilities in Canada. We want to unerstand how to classify outliers in solar production data. The data is collected hourly across several years from 2015 through April 2023. For our analysis, we will subset to data starting in 2021 and for the largest facility.

First, let's use reinforcement learnging, namely Q-learning from OpenAI Gym to diagnose for time series anomaly detection. Using conservative values for the Hyperparameters to reduce the number of diagnosed outliers yields a contamination rate of 0.52% (50 out 9,604). 

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/anomaly_ts_q_learning.png" width=50% height=50%>

Next, we can compare the Q-Learning results to the results from an Isolation Forest. We can define several features for this model as shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/isolation_forest_features.png" width=50% height=50%>

Next, we can estimate the contamination by using 3 standard deviations from the mean for each feature. The estimated contamination rate is very similar at 0.47%!

After training the Isolation Forest model using this estimated outlier rate, the result is shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/anomaly_ts_isolation_forest.png" width=50% height=50%>

When comparing models, the outlier rate, as mentioned, is very similar. However, none of the outliers overlap between models. This warrants model tuning and EDA so stay tuned for more!

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/model_anomaly_summary.png" width=50% height=50%>