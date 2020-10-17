import json
from logger.logger import logger

global seed
seed = 0

global configuration
configuration = {

    "DEBUG":    True,

    # These paths are relative to the main directory
    "paths":        {
        # 'data_root':    "data/SemEval14/",
        'data_root':    "data/SemEval16/",
        # 'data_root': "data/MAMS_ACSA/",
        # 'data_root': "data/FourSquared/",
        # 'data_root':  "data/SamsungGalaxy/",
        "output":    "output/",
        "dataset":  "train.xml",
        "saved_graph":    "SemEval16_train_graph.bin",
        "dataframe": "SemEval16_dataframe.csv",
        "label_text_to_label_id": "SemEval16_label_text_to_label_id.json",
    },

    "data":         {
        "dataset":     {
            'name': 'SemEval16'
            # 'name':   'MAMS_ATSA'
            # 'name':       'SamsungGalaxy'
            # 'name':     'FourSquared'
        },
        "trainval_test_split": 0.3,
        "train_val_split":  0.3,
        "min_label_occurences": 0
    },

    "model": {
        'in_dim': 300,
        'hidden_dim': 150,
        'num_heads': 2,
    },

    "training":        {
        "seed":                 23,
        "epochs":               2,
        "create_dataset":       False,
        "dropout":              0.2,
        "train_batch_size":     30,
        "val_batch_size":       60,
        "test_batch_size":      60,
        "early_stopping_patience": 6,
        "early_stopping_delta": 0,
        "optimizer":            {
            "optimizer_type": "adam",
            "learning_rate":  3e-4,
        },
        "threshold": 0.5,
    },

    "embeddings":   {
        'embedding_file': 'glove-twitter-25',
    },

    "hardware": {
        "num_workers": 16
    }
}


class Config(object):
    """ Contains all configuration details of the project. """

    def __init__(self):
        super(Config, self).__init__()

        self.configuration = configuration

    def get_config(self):
        return self.configuration

    def print_config(self, indent=4, sort=True):
        """ Prints the config. """
        logger.info("[{}] : {}".format("Configuration",
                                       json.dumps(self.configuration,
                                                  indent=indent,
                                                  sort_keys=sort)))

    @ staticmethod
    def get_platform():
        """ Returns dataset path based on OS.

        :return: str
        """
        import platform

        if platform.system() == 'Windows':
            return platform.system()
        elif platform.system() == 'Linux':
            return platform.system()
        else:  # OS X returns name 'Darwin'
            return "OSX"

    @ staticmethod
    def get_username():
        """
        :returns the current username.

        :return: string
        """
        try:
            import os
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name

        except Exception:
            import getpass

            username = getpass.getuser()
        finally:
            username = os.environ.get('USER')

        return username


config_cls = Config()
config_cls.print_config()

global platform
platform = config_cls.get_platform()
global username
username = config_cls.get_username()
