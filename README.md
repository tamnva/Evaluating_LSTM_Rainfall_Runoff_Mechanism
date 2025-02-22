## Evaluating LSTM Rainfall Runoff Mechanism

#### 1. Scenario simulations with SWAT and LSTM

- The source code for scenario simulation with SWAT model: lstm_swat/scenario_simulation_swat.R 
- The source code for scenario simulation with LSTM model: lstm_swat/scenario_simulatoin_lstm.py
- LSTM_Q and LSTM_Q models and model structures

Additional data needed to run the R and Python Script, which could be provided upon request.


#### 2. The trained LSTM rainfall runoff for 1455 catchments ([median NSE = 0.7 for the test period](https://github.com/tamnva/Evaluating_LSTM_Rainfall_Runoff_Mechanism/blob/master/lstm_camel_de/results/nse_test_period.csv))

###### 2.1. Training configurations are in the file (details can be found in this file  [lstm_camel_de/config.yml](lstm_camel_de/config.yml))

- Training period: 1980-2000
- Validation period: 2001-2010
- Testing period: 2011-2020
- Input time series data: 
  - precipitation_mean
  - radiation_global_mean
  - temperature_min
  - temperature_max
- Input catchment attributes
  - p_mean
  - p_seasonality
  - area
  - elev_mean
  - clay_0_30cm_mean
  - sand_0_30cm_mean
  - silt_0_30cm_mean
  - artificial_surfaces_perc
  - frac_snow
- Model structure: hidden size = 256, one LSTM layer, Linear model head
- Data for running this model could be downloaded from https://doi.org/10.5281/zenodo.13320514. The data need to be converted to the format as described at  https://hydroecolstm.readthedocs.io/en/latest/data.html#input-data-format



###### 2.2 Run catchment DE110000 with input precipitation = 0 

- The model performance for the test period (2011-2020) with original data is high (NSE = 0.92)
- Below is the simulated streamflow for this catchment with precipitation = 0 (results can be reproduced using this script [lstm_camel_de/load_run_trained_model.py](lstm_camel_de/load_run_trained_model.py))

<p align="center">
  <img src="https://github.com/tamnva/Evaluating_LSTM_Rainfall_Runoff_Mechanism/blob/master/lstm_camel_de/results/simulated_streamflow_DE110000_no_rainfall.png" width=80% title="hover text">
</p>
