## Data Preparation
Our data source is solar panel production from major solar facilities in Canada. We want to understand how to classify outliers within this solar production data. The data is collected hourly across several years from 2015 through April 2023. For our analysis, we will reduce the scope to data starting in 2021 and for one plant. Also since nightly hours do not have solar production, we will aggregate the data by date. A time-series view of this plant's solar kWh production is shown below.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/whitehorn_ts.png" width=100% height=100%>

## Reinforcement Learning for Outlier Detection
First, let's use reinforcement learning, namely Q-learning from OpenAI Gym, to diagnose time series anomalies. Using conservative values for the Hyperparameters to reduce the number of diagnosed outliers yields a contamination (or, outlier) rate of 2%. 

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/ql_anomaly.png" width=100% height=100%>

## Isolation Forest for Outlier Detection
Next, we can compare the Q-Learning results to the results from an Isolation Forest model. We can define several features for this model considering a window of 14 days. These metrics are shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/isolation_forest_features1.png" width=25% height=25%>

Next, we can estimate the contamination by using 1.75 standard deviations from the mean for each feature, then averaging the number of outliers over total oberservations across all metrics. The estimated contamination rate is very similar at 2.6%. But for the Isolation Forest, we will use the exact outlier rate from the Q-Learning result.

After training the Isolation Forest model using the outlier rate from the Q-Learning model (2.17%), the result is shown below:

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/if_anomaly.png" width=100% height=100%>

## Rolling Mean for Outlier Detection

Using the rolling mean is the most logical metric we should consider for a more simplified model. For this analysis, we can define the window to be the previous 7 days. Then in order to yield an outlier ratio close to the results from the other two models, we can set the outlier threshold to be 1.75*(standard deviation). That is, if an observed value differs from the past 7-day mean by more than 1.75 times the standard deviation (of the rolling mean), then we flag the value as an outlier.

## Comparing All 3 Models

When comparing models, the outlier rates are very similar. Below is a chart showing all three outlier models compared, then also a table showing the percent of outliers in each model bucket.

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/ts_model_comp.png" width=100% height=100%>

<img src="https://github.com/mwheeler235/ts-anamoly-detection-q-learning/blob/main/img/model_anomaly_summary2.png" width=90% height=90%>

## Observations
* For the Isolation Forest's outliers, about half of them are picked up by the Rolling Mean model. Only one from the Isolation Forest overlapped with the Q-Learning outlier bucket.
* The majority of the Q-Learning outliers were not flagged by the other two models, indicating that perhaps further tuning of the RL model is required to approach more convergence.


## Conclusion
This is an interesting comparison on daily solar production data. However, looking at the raw hourly data for outliers would yield more actionable insights as this would allow an organization to make more near-real-time decisions. Next steps should include modeling at the hourly level with considerations on how to handle (or drop) nightly hours. Additionally, other plants should be tested to understand how each model performs at different locations.
