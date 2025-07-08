from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='FDL_pytorch',
    version='1.0',
    description='Frequency Distribution Loss (FDL) for misalignment data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['FDL_pytorch'],
    author='Zhangkai Ni, Juncheng Wu, Zian Wang',
    author_email='zkni@tongji.edu.cn',
    install_requires=["torch>=1.0"],
    url='https://github.com/eezkni/FDL',
    keywords = ['pytorch', 'loss', 'image transformation','misalignment'], 
    platforms = "python",
    license='MIT',
)