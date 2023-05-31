import yaml


def read_config_file(yaml_path: str):
    data = {}
    with open(yaml_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data
