import argparse
import os
import yaml

# this module is used to to parse flplans in preparation of building docker containers
# here we have a copy of load_yaml from tfedlrn.utils (trying to avoid instalation of tfedlrn)
def load_yaml(path):
    """Parses the Federation Plan (FL plan) from the YAML file.

    Args:
        path: The directory path for the FL plan YAML file.

    Returns:
        A YAMLObject with the Federation plan (FL Plan)

    """
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan


def get_until_none(key_list, _dict):
    """Return string version of info found in federation (FL) plan when following key list until key==None or end of key_list

    Args:
        key_list: A list of keys from the dictionary
        _dict: The dictionary to parse

    Returns:
        The string value of the dictionary for that key

    """
    result = _dict
    for key in key_list:
        if key is None:
            break
        else:
            result = result[key]
    return str(result)


def main(plan, key_list=[None]):

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = load_yaml(os.path.join(plan_dir, plan))

    info = get_until_none(key_list = key_list, _dict=flplan)

    # printing info to stdout
    print(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--key_list', '-kl', type=str, nargs="*")
    args = parser.parse_args()
    main(**vars(args))
