"""!@package config_parser
@brief Module creating a parser for default command line arguments

@details
The default config_parser object can be further customized as needed.

For function details see the function documentation:
- config_parser.py

@author Mate Balogh
@date 2024-08-06
"""

import argparse
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def default_parser(description):
    """!@brief creates a parser for default command line arguments
    @details
    The default arguments are: <br>
    -i, --input-file: ARW file to be processed <br>
    -o, --output-suffix: string to be appended to all output filenames <br>
    -c, --config-file: configuration file defining initial hyperparameters <br>
    -ow, --overwrite: overwrite existing files <br>

    @param description (str): text that is displayed in the -h help output

    @return config parser: command line argument parser
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-file",
        help="ARW file to be processed",
        required=False,
        default=root_dir + "/data/DSC06639.ARW",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-suffix",
        help="This string is appended to all output filenames",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Configuration file defining api keys",
        required=False,
        default=root_dir + "/config/default-config.ini",
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        help="Overwrite existing files",
        required=False,
        action="store_true",
    )
    return parser


def remove_argument(parser, arg):
    """!@brief remove an argument from an already existing command line parser
    @param parser (config parser): command line argument parser to remove from
    @param arg (str): parameter to remove, this is the long -- handle, with _s
    instead of -s (eg. random_seed instead of random-seed)
    """
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return
