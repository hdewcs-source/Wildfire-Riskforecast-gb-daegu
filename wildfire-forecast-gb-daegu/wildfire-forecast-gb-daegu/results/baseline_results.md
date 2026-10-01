## FFDRI_new recomputed from the published coefficients

| Statistic | Recomputed | Report |
| --- | --- | --- |
| count | 32880 | 32880 |
| mean | 14.969 | 14.970 |
| std | 5.229 | 5.229 |
| min | 1.807 | 1.807 |
| 25 percent | 11.256 | 11.257 |
| median | 14.915 | 14.916 |
| 75 percent | 18.750 | 18.751 |
| max | 30.126 | 30.128 |

Pearson correlation between FFDRI_new and the national FFDRI over all rows: 0.808. 90th percentile (start of the Extreme grade): 21.89.

## Season definition in the dataset

- `season == fall` contains months 7 to 12
- `season == spring` contains months 1 to 6

## 7-day forecast baselines, test year 2024

Protocol: 14-day lookback, 7-day horizon, train before 2023, test 2024, one evaluation per season value of the dataset. RMSE and MAE are in FFDRI_new units.

| Season | Method | t+1 RMSE | t+1 MAE | t+1 corr | 1-7 day RMSE | 1-7 day MAE | test sequences |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fall | Persistence | 2.328 | 1.689 | 0.897 | 2.856 | 2.174 | 2460 |
| fall | Ridge on 14 lags | 2.071 | 1.612 | 0.915 | 2.337 | 1.912 | 2460 |
| spring | Persistence | 2.466 | 1.852 | 0.886 | 3.195 | 2.514 | 2430 |
| spring | Ridge on 14 lags | 2.177 | 1.687 | 0.907 | 2.597 | 2.123 | 2430 |
