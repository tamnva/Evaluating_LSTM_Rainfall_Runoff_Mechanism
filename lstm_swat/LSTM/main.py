
from hydroecolstm.model_run import run_config
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.create_model import create_model
from hydroecolstm.interface.utility import write_yml_file
from hydroecolstm.utility.evaluation_function import EvaluationFunction
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from hydroecolstm.utility.plot import plot

# Global settings
working_dir = "C:/Users/nguyenta/Documents/Manuscript/lstm_swat"

#-----------------------------------------------------------------------------#
#                                    LSTM Q                                    #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config(Path(working_dir, "data/LSTM/config.yml"))
config['output_directory'] = [working_dir + "/data/LSTM/LSTM_Q"]
config['target_features'] = ['q_mm']

# Create model and train from config 
model, data, best_config = run_config(config)

data["loss_epoch"].plot()
plt.show()

# Evaluate the model and transform to normal scale
data['y_train_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_train_scale"]))
data['y_valid_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_valid_scale"]))
data['y_test_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_test_scale"]))

objective = EvaluationFunction(config['eval_function'], 
                               config['warmup_length'],
                               data['y_column_name'])
objective(data['y_train'], data['y_train_simulated'])
objective(data['y_valid'], data['y_valid_simulated'])
objective(data['y_test'], data['y_test_simulated'])

# Visualize valid and test data
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), target_feature=target)
        p.show()
        
# Save all data and model state dicts to the output_directory
torch.save(data, Path(config["output_directory"][0], "data.pt"))
torch.save(model.state_dict(), 
           Path(config["output_directory"][0], "model_state_dict.pt"))
write_yml_file(config = best_config,
               out_file=Path(config["output_directory"][0], "best_config.yml"))

#-----------------------------------------------------------------------------#
#                                  LSTM Q_ET                                  #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config(Path(working_dir, "data/LSTM/config.yml"))
config['output_directory'] = [working_dir + "/data/LSTM/LSTM_Q_ET"]
config['target_features'] = ['eta_mm', 'q_mm']

# Create model and train from config 
model, data, best_config = run_config(config)

data["loss_epoch"].plot()
plt.show()

# Evaluate the model and transform to normal scale
data['y_train_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_train_scale"]))
data['y_valid_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_valid_scale"]))
data['y_test_simulated'] = data["y_scaler"].inverse(
    model.evaluate(data["x_test_scale"]))

objective = EvaluationFunction(config['eval_function'], 
                               config['warmup_length'],
                               data['y_column_name'])
objective(data['y_train'], data['y_train_simulated'])
objective(data['y_valid'], data['y_valid_simulated'])
objective(data['y_test'], data['y_test_simulated'])

# Visualize valid and test data
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), target_feature=target)
        p.show()
        
# Save all data and model state dicts to the output_directory
torch.save(data, Path(config["output_directory"][0], "data.pt"))
torch.save(model.state_dict(), 
           Path(config["output_directory"][0], "model_state_dict.pt"))
write_yml_file(config = best_config,
               out_file=Path(config["output_directory"][0], "best_config.yml"))
