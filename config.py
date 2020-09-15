import json
from logger.logger import logger

global seed
seed = 0

global configuration
configuration = {

    "DEBUG":    True,

    "data":         {
        "dataset":     {
            'name':   'SemEval16'
        },
        "trainval_test_split": 0.7,
        "train_val_split":  0.7,
        "show_stat":  False
    },

    "model": {
        'in_dim': 1,
        'hidden_dim': 4,
        'num_heads': 2,
    },

    "training":        {
        "create_dataset":       True,
        "dropout":              0.2,
        "max_epochs":           10,
        "train_batch_size":     30,
        "val_batch_size":       15,
        "test_batch_size":      15,
        "optimizer":            {
            "optimizer_type": "adam",
            "learning_rate":  3e-4,
            "lr_decay":       0,
            "weight_decay":   0,
            "momentum":       0.9,
            "dampening":      0.9,
            "alpha":          0.99,
            "rho":            0.9,
            "centered":       False
        },
    },

    "embeddings":   {
        'embedding_file': 'glove-twitter-25',
        'emb_dim':        100,
    },

    "gnn_params":   {
        "padding":     1,
        "stride":      1,
        "kernel_size": 1,
        "bias":        True,
    },

    # These paths are relative to the main directory
    "paths":        {
        # for saving plots, f1_score
        "data_root":          "data/SemEval16/",
        "output":    "output/",
        "log":       "logs/",
        "dataset":  "train.xml",
        "saved_graph":    "SemEval_train_graph.bin"
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

        except Exception as e:
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
