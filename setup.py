from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ZERN',
    version='0.9.1',
    description='Fast computation of Zernike polynomials',
    long_description=long_description,
    url='https://github.com/AlvaroMenduina/ZERN',
    author_email='alvaro.menduinafernandez@gmailcom',
    classifiers=['Development Status :: 3 - Alpha'],
    packages=find_packages(),
    install_requires=[
        'numpy>=1.0',
        'matplotlib'  # Specify the desired numpy version or remove for any version
        # Add other dependencies as needed
    ]
)