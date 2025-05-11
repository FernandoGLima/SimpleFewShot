from setuptools import setup, find_packages

setup(
    name='simplefsl', 
    version='0.1.0',  
    packages=find_packages(),
    install_requires=[
        'torch==2.7.0',
        'pandas==2.2.3',
        'torchvision==0.22.0',
        'scikit-learn==1.6.1',
        'timm==1.0.15',
        'qpth==0.0.18',
    ],
    author='Victor Nascimento Ribeiro',
    author_email='victor_nascimento@usp.br',
    description='A Python Library for Few-Shot Learning Models',
    url='https://github.com/victor-nasc/SimpleFewShot',
    license="MIT",  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
