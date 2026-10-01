# Wildfire Risk Forecasting for Gyeongbuk and Daegu

A coordinate-based pipeline that builds daily environmental time series for 15 wildfire-prone locations in Gyeongbuk and Daegu (2019 to 2024), defines a regional risk index called FFDRI_new, and forecasts it 7 days ahead with GRU and LSTM models. The project was done for an Industrial Mathematics course at Sogang University (fall 2025).

Team: Hyun Daeho (team leader), Ahn Gaeun, Jang Donghwan. The team repository is [HWANDJANG/Daegu-Gyeongbuk-Wildfire-Forecast](https://github.com/HWANDJANG/Daegu-Gyeongbuk-Wildfire-Forecast). The final report in Korean is in `docs/final_report_kr.pdf`.

## Pipeline

1. **Data collection with Google Earth Engine.** ERA5-Land weather (temperature, humidity, wind, precipitation), MODIS NDVI, ERA5 solar radiation, and SRTM elevation and slope are extracted by coordinate and date. Forest type comes from the Korea Forest Service forest-type map.
2. **National index components.** DWI (weather), FMI (forest type), TMI (terrain), and a day weight give the national index: FFDRI = (7 DWI + 1.5 FMI + 1.5 TMI) x day weight.
3. **Coordinate selection.** 15 representative locations, 5 each for dry, windy and steep-slope risk types, chosen from past fire locations and from the steepest and windiest forest areas.
4. **FFDRI_new.** A Ridge regression of the national FFDRI on DWI, FMI, TMI, NDVI and solar radiation, with coefficients converted back to the original scale:
   `FFDRI_new = 5.3012 + 6.6232 DWI + 0.7657 FMI + 1.0994 TMI + 0.0906 sunlight - 10.6796 NDVI`
5. **Forecast models.** 14-day lookback, 7-day multi-step output, one model per season, trained on data before 2023, validated on 2023 and tested on 2024.
6. **Risk grades and map.** Quantile-based grades of FFDRI_new (Low, Moderate, High, Very High, Extreme) and a map of the forecasts at five mountains.

## Repository layout

```
data/lstm_dataset.csv        daily series, 15 coordinates x 2192 days (32,880 rows)
data/wildfire_dataset.csv    1,463 samples (598 fire, 865 non-fire) used for the classification experiments
models/                      saved GRU and LSTM models, see models/README.md
notebooks/                   data pipelines, clustering, first model experiments, target trials
src/total_code.py            all notebook code merged into one script, sections separated by comments
src/add_ffdri_new.py         adds the FFDRI_new column that the training code expects
analysis/verify_and_baselines.py   recomputation of FFDRI_new and simple forecast baselines
results/baseline_results.md  output of the script above
docs/final_report_kr.pdf     final course report
```

`src/total_code.py` is not a single runnable script. It is the notebook code concatenated, so it still contains Colab commands such as `!pip` and file-upload prompts, and each stage is marked with a comment. Set your own Earth Engine project in place of `YOUR_GEE_PROJECT_ID` before running the extraction stages.

## Results

### Reported by the team

The report gives these test results for 2024 (Table 2 of the report). The printouts in the slides show different numbers for the same models, so both are listed. The models were not re-run for this repository.

| Season | Model | t+1 RMSE | t+1 MAE | t+1 corr | 1-7 day RMSE | Source |
| --- | --- | --- | --- | --- | --- | --- |
| spring | GRU | 2.09 | 1.80 | 0.91 | 2.07 | report table |
| spring | GRU | 2.27 | 1.90 | 0.90 | 2.48 | slide printout |
| spring | LSTM | 2.50 | 2.11 | 0.89 | 2.52 | report table |
| fall | GRU | 2.18 | 1.85 | 0.91 | 2.08 | report table |
| fall | GRU | 2.12 | 1.72 | 0.91 | 2.29 | slide printout |
| fall | LSTM | 2.38 | 2.02 | 0.90 | 2.41 | report table |

### Simple baselines, recomputed here

Same protocol (14-day lookback, train before 2023, test 2024), with the `season` column of the dataset. Persistence repeats the last observed value. Ridge is a linear model on the 14 past FFDRI_new values.

| Season | Method | t+1 RMSE | t+1 MAE | t+1 corr | 1-7 day RMSE |
| --- | --- | --- | --- | --- | --- |
| spring | Persistence | 2.466 | 1.852 | 0.886 | 3.195 |
| spring | Ridge on 14 lags | 2.177 | 1.687 | 0.907 | 2.597 |
| fall | Persistence | 2.328 | 1.689 | 0.897 | 2.856 |
| fall | Ridge on 14 lags | 2.071 | 1.612 | 0.915 | 2.337 |

A linear model on the past values of the index reaches about the same t+1 error and correlation as the recurrent models. The recurrent models are not shown to be better than this linear baseline on this test year.

## Known issues

These were found while organizing the repository. They are listed so that the numbers above can be read correctly.

- **FFDRI_new is fitted to the national FFDRI, not to fire outcomes.** It correlates 0.81 with the national index over all rows. The report's attempts to model fire occurrence and burned area directly gave R2 near or below 0, so the index has not been validated against actual fires.
- **Season definition.** The report describes spring as February to April and fall as October to December. The `season` column in `data/lstm_dataset.csv`, which the training code uses, splits the year into January to June and July to December, and the test sequence counts (162 per coordinate) match that definition.
- **Model description.** The report describes a univariate input of shape (14, 1). The saved GRU models take 11 features with shape (14, 11). The saved LSTM models take a univariate input of 30 days and have a different architecture from the one in the report.
- **Report table.** The 1-7 day RMSE of the GRU in Table 2 is lower than its t+1 RMSE, which is unusual for a multi-step forecast. The slide printouts show the expected ordering.
- **Scaler.** The fitted scalers were not saved, and the forecasting section of the code refits one on the first 80 percent of windows, which differs from the training split.
- **Window continuity.** Training windows are built from consecutive rows of a season subset without checking that the dates are adjacent, so some training windows can span the gap between years. The 2024 test windows are not affected.
- **Case studies.** The forecasts for 2025-04-28 and 2025-12-03 are single dates, and the observed values for the following 7 days are not shown. The RMSE of 2.1 quoted for the 2025-04-28 case is the 2024 test RMSE. In the 2025-04-28 table, values of 19.26 to 19.38 are labelled with grades that do not match the quantile cut points of the report (Very High starts at 18.75 and Extreme at 21.89).
- **Mountain comparison.** For four of the five mountains the D0 value lies within 0.11 of each other. The model has no location, access or population inputs, so the report's explanations for the differences are interpretation and not model output.
- **Scale.** 15 coordinates, six years and one test year. Daily values are strongly autocorrelated.

## Reproduce the checks

```bash
pip install -r requirements.txt
python analysis/verify_and_baselines.py --data data/lstm_dataset.csv --out results
python src/add_ffdri_new.py data/lstm_dataset.csv data/lstm_dataset_with_ffdri_new.csv
```

## Data sources and terms

ERA5-Land and ERA5 (Copernicus), MODIS NDVI, SRTM elevation, the Korea Forest Service forest-type map and wildfire damage records. The terms of the original sources apply. Check the redistribution terms of the forest-type map, the wildfire records and any geocoding service before reusing the data files.
