import argparse
import os

from tfedlrn import load_yaml
from setup_logging import setup_logging


def get_until_none(key_list, _dict):
    # return string version of info found in flplan when following key list until key==None or end of key_list
    result = _dict
    for key in key_list:
        if key is None:
            break
        else:
            result = result[key]
    return str(result)
    

def main(plan, first_key, second_key, third_key, logging_config_fname, logging_default_level):
    
    setup_logging(path=logging_config_fname, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    key_list = [first_key, second_key, third_key]

    flplan = load_yaml(os.path.join(plan_dir, plan))

    info = get_until_none(key_list = key_list, _dict=flplan)

    # printing info to stdout
    print(info)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--first_key', '-fk', type=str, default=None)
    parser.add_argument('--second_key', '-sk', type=str, default=None)
    parser.add_argument('--third_key', '-tk', type=str, default=None)
    parser.add_argument('--logging_config_fname', '-lc', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))