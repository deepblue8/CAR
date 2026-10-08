import torch
import yaml
from GARMain import GARMain
from pprint import pprint


def init_Config():
    cfgPath = 'config/config.yaml'

    with open(cfgPath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['trainOrVal'] = 'train'

    return config

def main():
    cfg = init_Config()
    pprint(cfg)

    garMain = GARMain(cfg)
    if cfg['trainOrVal'] == 'train':
        print('Start Train....................')
        garMain.train()
    elif cfg['trainOrVal'] == 'val':
        print('Start Val....................')
        garMain.val()
    else:
        print('Start Train and Val....................')

if __name__ == '__main__':
    main()
