import argparse
import os
import yaml

# this module is used to to parse flplans in preparation of building docker containers
# here we have a copy of load_yaml from tfedlrn.utils (trying to avoid instalation of tfedlrn)
def load_yaml(path):
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan


def get_until_none(key_list, _dict):
    # return string version of info found in flplan when following key list until key==None or end of key_list
    result = _dict
    for key in key_list:
        if key is None:
            break
        else:
            result = result[key]
    return str(result)
    

def main(plan, first_key, second_key, third_key):
    
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
    args = parser.parse_args()
    main(**vars(args))