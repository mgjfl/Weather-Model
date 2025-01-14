import yaml
import sys
import os
import pathlib
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def load_config(self):
        """Load configuration from a YAML file."""
        with open(os.path.join(HOME_DIR, "Results", "Configurations", self.config_path), 'r') as file:
            self.config = yaml.safe_load(file)

    def save_config(self, output_path):
        """Save configuration to a YAML file."""
        with open(os.path.join(HOME_DIR, "Results", "Configurations", output_path), 'w') as file:
            yaml.dump(self.config, file)

    def get_data_config(self):
        """Retrieve data configuration."""
        return self.config.get('data', {})

    def get_model_config(self):
        """Retrieve model configuration."""
        return self.config.get('model', {})

    def get_training_config(self):
        """Retrieve training configuration."""
        return self.config.get('training', {})

    def get_output_config(self):
        """Retrieve output configuration."""
        return self.config.get('output', {})

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