from setuptools import setup, find_namespace_packages

setup(name='_20220409_Conv',
      packages=find_namespace_packages(include=['base_conv', 'base_conv**','test_module']),
      version='0.0.1')

'''
推測: 把某個功能模組化。

'''