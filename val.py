import torch
import yaml
from GARMainVal import GARMain
from pprint import pprint

def init_Config():
    cfgPath = 'config/config.yaml'
    with open(cfgPath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['trainOrVal'] = 'val'

    return config

def main():
    cfg = init_Config()

    garMain = GARMain(cfg)
    print('Start Val....................')
    garMain.val()

if __name__ == '__main__':
    main()
