import os,sys
downpath = '//home/yoyoteng/scisoft/data/numerai'

sys.path.append(
r'//home/yoyoteng/scisoft/NumerAPI'
)
from numerapi.numerapi import NumerAPI

import yaml

with open('cfg.yml','r') as f:
    cfg=yaml.load(f.read())
    
inst = NumerAPI(**cfg)

#if __name__ == '__main__':
#    inst.download_current_dataset(dest_path=downpath,unzip=True)
