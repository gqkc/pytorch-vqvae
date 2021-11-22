import setuptools

setuptools.setup(
    name='pytorch_vqvae',
    version='0.0.1',
    author="QUISPE guillaume",
    author_email="guillaume.quispe@gmail.com",
    description="vqvae implementation",
    url="https://github.com/gqkc/pytorch-vqvae.git",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'numpy'], )
