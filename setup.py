from setuptools import setup, find_packages

setup(
    name='projective_geometry',
    version='1.0.0',
    author='Andre Aichert',
    author_email='aaichert@gmail.com',
    description='A collection of numpy-based utilities for projective geometry of real two- and three-space, including homogeneous coordinates of point, lines and planes, Plücker coordinates and projection matrices.',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
)