from setuptools import setup, find_packages

setup(
    name='infantnetwork',
    version='0.1.0',
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'pandas',
        'numpy'
    ],
    author='Daniel Helkey',
    author_email='dhelkey@stanford.edu',
    description='A Python package to compute neonatal transfer networks and describe their shape and structure.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dhelkey/infantnetwork',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
