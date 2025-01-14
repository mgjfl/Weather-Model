import wandb
import os
import json
import yaml
import datetime

class DataSaver:
    
    def __init__(self, config, run_number = None):
        
        
        self.run = wandb.init(
            project=config['output']['project_name'],
            name=config['output']['run_name'] + (("_" + str(run_number)) if run_number is not None else ""),
            config=config
        ) if config["output"]["wandb"] else None
        
        self.config = config
        self.results = dict()
        self.metrics = config["output"]["metrics"]
        self.run_number = run_number
        
        for metric in self.metrics:
            if self.run:
                wandb.define_metric(metric, summary = self.wandb_summary(metric))
            self.results[metric] = list()
            
    def wandb_summary(self, metric):
        
        if "loss" in metric:
            return "min"
        if "acc" in metric:
            return "max"
        
        return "last"
            
    def log_metrics(self, log_dict):
        for (metric, value) in log_dict.items():
            if metric in self.metrics:
                self.results[metric].append(value)

        if self.run:
            self.run.log(log_dict)
            
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
        
        
    def end_run(self):
        
        if self.run:
            self.run.finish()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results["timestamp"] = timestamp
        
        # Set up the run dir
        base_path = self.get_base_dir()
        run_dir = self.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config and metrics
        self.save_metrics(run_dir)
        self.save_config(base_path)

