from setuptools import setup, find_packages

setup(
    name            = 'multiresunet',
    version         = '0.1',
    description     = 'MultiResUNet implementation in PyTorch; MultiResUNet: Rethinking the U-Net Architecture for Multimodal',
    author          = 'Younghan Kim',
    author_email    = 'godppkyh@mosqtech.com',
    install_requires= [],
    packages        = find_packages(),
    python_requires = '>=3.6'  
)
