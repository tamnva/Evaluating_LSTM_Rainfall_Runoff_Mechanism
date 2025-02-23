

from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.create_model import create_model
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import os
import pandas as pd


# Set up working directory
os.chdir("C:/lstm_camel_de")

#-----------------------------------------------------------------------------#
#                          Load the trained model                             #
#-----------------------------------------------------------------------------#
config = read_config("config.yml")
model = create_model(config)
model.load_state_dict(torch.load(Path(config["output_directory"][0], 
                                      "model_state_dict.pt")))
data = torch.load(Path(config["output_directory"][0], "data.pt"))

# To execute the code below, please download the camel_de.zip file and find
# the file camels_de/timeseries/CAMELS_DE_hydromet_timeseries_DE110000.csv
# then put this file in to the folder .lstm_camel_de/input_data
# Link to camel_de.zip https://zenodo.org/records/13837553

input_data = pd.read_csv("input_data/CAMELS_DE_hydromet_timeseries_DE110000.csv",
                         parse_dates=True, index_col=[0])

# Our model just need precipitation, radiation, tempearture, and catchment 
# attributes (which can be found in the camels_de extracted above)

x_input = pd.DataFrame({
    'precipitation_mean':input_data['precipitation_mean'],
    'radiation_global_mean':input_data['radiation_global_mean'],
    'temperature_min':input_data['temperature_min'],
    'temperature_max':input_data['temperature_max'],
    'p_mean':2.97,
    'p_seasonality':0.01,
    'area':763.07,
    'elev_mean':822.25,
    'clay_0_30cm_mean':28.89,
    'sand_0_30cm_mean':24.79,
    'silt_0_30cm_mean':41.18,
    'artificial_surfaces_perc':7.84,
    'frac_snow':0.12})



# Convert x input to torch tensor (float32)
x_input_tensor = torch.tensor(x_input.values, dtype=torch.float32)

# Now set rainfall to zero (in the first column of input)
x_input_tensor[:,0] = 0.0

# Scale/Transform input  (Note: input to data x_scaler is a dict)
x_input_tensor_scale = data["x_scaler"].transform({"DE110000": x_input_tensor})


# Now simulate streamflow with rainfall = 0
Q_sim = data["y_scaler"].inverse(model.evaluate(x_input_tensor_scale))

# Plot the results
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))
ax.plot(input_data.index, Q_sim["DE110000"], linewidth = 1)
ax.set_ylabel(r'Streamflow (mm/day)')
ax.set_title("Simulated streamflow for catchment DE110000 with rainfall = 0")
#plt.savefig('results/simulated_streamflow_DE110000_no_rainfall.png')
plt.show()

