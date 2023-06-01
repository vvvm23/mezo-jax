from setuptools import setup

setup(
   name='mezo_jax',
   version='0.0.1',
   description='JAX Library implementing MeZO finetuning',
   license='Apache 2.0',
   long_description=open('README.md', 'r').read(),
   author='Alex McKinney',
   author_email='alex.f.mckinney@gmail.com',
   url='https://github.com/vvvm23/mezo-jax',
   packages=['mezo_jax'],
   install_requires=['wheel'],
)
