from pathlib import Path
from setuptools import find_packages, setup
dependencies = ['matplotlib', 'numpy', 'scipy', 'astropy']


# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hapa',
    packages=find_packages("src"),
    version='0.0.1',
    description='Halo Package',
    author='Gaétan Facchinetti',
    author_email='gaetanfacc@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=dependencies,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)