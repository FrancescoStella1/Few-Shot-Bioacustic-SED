from configparser import ConfigParser
import platform


if platform.system() == "Windows":
    CONFIG_FILE = "config/config_windows.ini"
else:
    CONFIG_FILE = "config/config_linux.ini"


def create_config():
    parser = ConfigParser()
    parser.read(CONFIG_FILE)
    return parser


CONFIG = create_config()