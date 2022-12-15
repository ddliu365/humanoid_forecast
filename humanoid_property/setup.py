#!/usr/bin/env python
import configparser
from distutils.core import setup
import sys
import os
# while installing, get current folder path and write into config
current_path=sys.argv[0]
pathname = os.path.abspath(os.path.dirname(sys.argv[0]))
configPath = os.path.abspath(os.path.join(current_path, '../python/robot_properties_solo/config/package_config.ini'))
config = configparser.ConfigParser()
config['Path'] = {'package_path': pathname}
with open(configPath, 'w') as configfile:
      config.write(configfile)

setup(name='robot_properties_solo',
      version='1.0',
      description='humanoid_configuration',
      author='ddliugit',
      author_email='ddliu@nyu.edu',
      url='https://www.python.org/sigs/distutils-sig/',
      packages = ['robot_properties_solo'],
      package_dir = {'': 'python'},
      package_data = {'': ['config/*']},
     )

