from setuptools import setup, find_packages

setup(
    name='ProjectiveGeometry23',
    version='0.1.0',
    description='Projective geometry in 2D and 3D with homogeneous and Plücker coordinates, projection matrices, and visualization.',
    author='Andre Aichert',
    author_email='aaichert@gmail.com',
    url='https://github.com/aaichert/ProjectiveGeometry23',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={
        'svg': ['svg_snip']
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)