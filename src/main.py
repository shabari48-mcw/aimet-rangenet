import argparse
import os
from utils.parse_json import parse_json
from quantize import inference as quantize
from contextlib import redirect_stdout
from rich.traceback import install
install()

def arguments(raw_args):
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Script for RangeNet model quantization')
    parser.add_argument('--config', help='model configuration to use', type=str, required=True)
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """ Run evaluations """
    args = arguments(raw_args)
    
    json_path = os.path.join(args.config)
    
    config= parse_json(json_path)
    
    print("HI")
      
    quantize(config['dataset'], config["Result"], config['pretrained_model'], config)
 

if __name__ == '__main__':
    with open('./log/quant.log', 'w') as f:
        with redirect_stdout(f):
            main()