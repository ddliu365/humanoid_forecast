#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

package_info = generate_distutils_setup()
package_info['packages'] = ['py_lqr']
package_info['package_dir'] = {'': 'python'}
package_info['install_requires'] = []

setup(**package_info)
