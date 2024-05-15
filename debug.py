import sys
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="conf/", config_name="test", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == '__main__':
    sys.argv.append('hydra.job.chdir=False')

    main()