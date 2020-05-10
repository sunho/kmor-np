from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kmor',
    version='1.0.7',
    description='K-means clustering with outlier removal numpy implementation',
    author='Sunho Kim',
    author_email='ksunhokim123@naver.com',
    packages=['kmor'],
    url='https://github.com/sunho/kmor-np',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy'],
    keywords=['kmor', 'anomality detection', 'cleaning', 'k-mean', 'data science'],
    python_requires='>=3'
)

