import argparse


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument("--config", type=str,
                        default="./config/config.yaml",
                        help="Yaml configuration file.")

    return parser.parse_args()
