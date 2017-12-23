import os,sys
import yaml

cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'cfg.yml')
with open(cfg_path,'r') as f:
    cfg=yaml.load(f.read())
downpath = cfg['downpath']
libpath = cfg['libpath']

for r in libpath:
	if r not in sys.path:
		sys.path.append(r)
