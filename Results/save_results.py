import wandb
import os
import json
import yaml
import datetime
import shutil
import errno
import stat
import torch
from collections import defaultdict

def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

class DataSaver:
    
    def __init__(self, config, run_number = None):
        
        
        self.run = wandb.init(
            project=config['output']['project_name'],
            name=config['output']['run_name'] + (("_" + str(run_number)) if run_number is not None else ""),
            config=config
        ) if config["output"]["wandb"] else None
        
        self.config = config
        self.results = defaultdict(list)
        self.metrics = config["output"]["metrics"]
        self.run_number = run_number
        
        for metric in self.metrics:
            if self.run:
                wandb.define_metric(metric, summary = self.wandb_summary(metric))
            
    def wandb_summary(self, metric):
        
        if "loss" in metric:
            return "min"
        if "acc" in metric:
            return "max"
        
        return "last"
            
    def log_metrics(self, log_dict):
        for (metric, value) in log_dict.items():
            self.results[metric].append(value)

        if self.run:
            self.run.log({k : v for k, v in log_dict.items() if k in self.metrics})
            
    def get_base_dir(self):
        base_path = os.path.join(
            "Results",
            "Runs",
            self.config["output"]["project_name"], 
            self.config["output"]["run_name"]
            )
        return base_path
        
    def get_run_dir(self):
        base_path = self.get_base_dir()
        
        if self.run_number is not None:
            base_path = os.path.join(base_path, f"Run_{self.run_number}")
            
        return base_path
        
    def save_metrics(self, output_dir):
        """Save metrics as a JSON file."""
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as file:
            json.dump(self.results, file, indent=4)

    def save_config(self, output_dir):
        """Save configuration as a YAML file."""
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)
            
    def save_model(self, output_dir, model):
        """Save PyTorch model in the TorchScript format."""
        torch.save(model, os.path.join(output_dir, 'trained_model.pt')) # Save
        
        
    def end_run(self, model):
        
        if self.run:
            self.run.finish()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results["timestamp"] = timestamp
        
        # Set up the run dir
        base_path = self.get_base_dir()
        run_dir = self.get_run_dir()
        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir, ignore_errors = False, onerror=handleRemoveReadonly)
            except Exception as e:
                print(e)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config and metrics
        self.save_metrics(run_dir)
        self.save_config(base_path)
        self.save_model(run_dir, model)

