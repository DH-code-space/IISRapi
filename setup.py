from setuptools import setup, find_packages
from pathlib import Path

def readmd():
    with open(Path("README.md"), "r", encoding="utf-8") as f:
        return f.read()
    
setup(
    name='IISRapi',
    version='2.1.4',
    packages=find_packages(),
    license='MIT',
    long_description=readmd(),
    long_description_content_type='text/markdown',
    install_requires=[
        'transformers',
        'flair',
        'torch'
    ]
)