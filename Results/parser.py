import yaml
import sys
import os
import pathlib
from collections import defaultdict
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)

def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', 
tuple_constructor)

class ConfigManager:
    """
    Manages the configuration files, containing model, data and training details.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def load_config(self):
        """Load configuration from a YAML file."""
        with open(os.path.join(HOME_DIR, "Results", "Configurations", self.config_path), 'r') as file:
            self.config = yaml.load(file, Loader=yaml.SafeLoader)

            # if self.config["data"]["type"] == "synthetic":
            #     self.config["output"]["project_name"] = f"synthetic_{self.config['data']['setting']}"
            self.config["output"]["project_name"] = self.config_path.split(os.sep)[0]

            self.config["output"]["run_name"] = f"{self.config['model']['arch']}"

            if "cov_class" in self.config['model'].keys():
                self.config["output"]["run_name"] = f"{self.config['model']['cov_class']}_" + self.config["output"]["run_name"]

            if "mean_class" in self.config['model'].keys():
                self.config["output"]["run_name"] = f"{self.config['model']['mean_class']}_" + self.config["output"]["run_name"]

    def save_config(self, output_path):
        """Save configuration to a YAML file."""
        with open(os.path.join(HOME_DIR, "Results", "Configurations", output_path), 'w') as file:
            yaml.dump(self.config, file)

    def get_data_config(self):
        """Retrieve data configuration."""
        data_config = self.config.get('data', {})
        return defaultdict(lambda: None, data_config)

    def get_model_config(self):
        """Retrieve model configuration."""
        model_config = self.config.get('model', {})
        return defaultdict(lambda: None, model_config)

    def get_training_config(self):
        """Retrieve training configuration."""
        training_config = self.config.get('training', {})
        return defaultdict(lambda: None, training_config)

    def get_output_config(self):
        """Retrieve output configuration."""
        output_config = self.config.get('output', {})
        return defaultdict(lambda: None, output_config)

# Example YAML template
def create_yaml_template():
    
    config_manager = ConfigManager("")
    config_manager.config = {
        'data': {
            'type': 'synthetic',                        # 'real' or 'synthetic'
            'parameters': {
                'w': 5,                                 # Number of weather variables
                'd': 10,                                # Number of spatial dimensions
                'n': 100                                # Number of datapoints
            },
            'global_seed': 42,                          # Global seed for reproducibility
            'n_runs' : 5,                               # Number of runs with random input
            'path': './data/',                          # Path for saving or loading data
            'gen_model' : 'PeriodicGaussianData',       # Generative data model
            'dataset' : 'NextDayForecastDataset'        # Dataset for different predictive tasks, e.g. NextDayForecastDataset or Past30DaysForecastDataset
        },
        'model': {
            'arch' : 'PNN',                             # Model architecture, 'PNN' or 'BNN'
            'type': 'FNO',                              # Model type, e.g., 'FNO', 'MLP', etc.
            'prob_model' : 'PeriodicGaussianData',      # Probabilistic model (only PNN); can be the same as generative data model
            'parameters' : {
                'hidden_channels': 64,                  # Embedding dimension
                'out_channels': 1,                      # Output dimension
                'n_modes': [16, 16],                    # Fourier modes
            }
        },
        'training': {
            'batch_size': 32,                           # Batch size
            'epochs': 3,                              # Maximum number of epochs for training
            'learning_rate': 0.001,                     # Learning rate
            'optimizer': 'Adam',                        # Optimizer class
            'loss_function': 'NLLLoss',                 # Loss function for training
            'early_stopping': {                         # Early stopping criterion
                'enabled': True,
                'patience': 10
            },
            'scheduler': {                              # Learning rate scheduler
                'type': 'StepLR',
                'parameters': {
                    'step_size': 20,
                    'gamma': 0.5
                }
            }
        },
        'validation': {
            'split_ratio': 0.2,                     # Fraction of data for validation
            'metrics': ['loss', 'accuracy']
        },
        'output': {
            'project_name': 'weather_prediction',
            'run_name': 'experiment_1',
            'wandb': True,
            'metrics' : ['train_loss', 'val_loss', 'lr']
        }
    }
    config_manager.save_config("config_template.yaml")




if __name__ == "__main__":

    # Create a YAML template
    create_yaml_template()