# Saved models

All four files are Keras 3.10 `.h5` files. Their input shapes were read from the model configuration.

| File | Input shape | Layers | Output |
| --- | --- | --- | --- |
| `gru_spring.h5` | (14, 11) | GRU(64, sequences) - GRU(32) - Dropout(0.2) - Dense(7) | FFDRI_new D+1 to D+7 |
| `gru_fall.h5` | (14, 11) | same as above | FFDRI_new D+1 to D+7 |
| `lstm_ffdri_new_spring.h5` | (30, 1) | LSTM(64) - Dropout(0.2) - Dense(64) - Dense(7) | FFDRI_new D+1 to D+7 |
| `lstm_ffdri_new_fall.h5` | (30, 1) | same as above | FFDRI_new D+1 to D+7 |

The 11 input features of the GRU models, in the order used by `src/total_code.py`:
`Tmean, RH, WSPD, TP_mm, sunlight_era5, DWI, FMI, TMI, NDVI, day_weight, FFDRI`.
The target `FFDRI_new` is not among the inputs.

The fitted `StandardScaler` objects were not saved. The scaler has to be rebuilt from the training
rows (dates before 2023-01-01, one scaler per season) before the models can be used on new data.
