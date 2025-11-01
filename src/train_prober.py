from wscgen.pipeline import PipelineConfig
from wscgen.training import ProberTrainingPipeline
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    cfg = PipelineConfig.load_yaml(args.config)
    ProberTrainingPipeline(cfg).run()

if __name__ == "__main__":
    main()