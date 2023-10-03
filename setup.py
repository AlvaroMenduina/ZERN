from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Load dependencies from requirements.txt
with open(path.join(here, 'requirements.txt'), 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='ZERN',
    version='1.0.0',
    description='Computing Zernike polynomials with Python',
    long_description=long_description,
    url='https://github.com/AlvaroMenduina/ZERN',
    author_email='alvaro.menduinafernandez@gmailcom',
    classifiers=['Development Status :: 3 - Alpha'],
    packages=find_packages(),
    install_requires=install_requires
)