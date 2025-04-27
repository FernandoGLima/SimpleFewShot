from setuptools import setup, find_packages

setup(
    name='simplefsl', 
    version='0.1.0',  
    packages=find_packages(),
    install_requires=[
        'torch',                  
        'pandas',              
        'torchvision',           
        'scikit-learn',     
        'timm',            
        'qpth',  
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
