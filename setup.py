from setuptools import setup

setup(
    name='cassandralin',
    version='0.0.1',
    packages=['cassandralin']
)

# numpy should be <= 1.23.5, otherwise GPy cannot be imported...
