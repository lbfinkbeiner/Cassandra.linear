from setuptools import setup

setup(
    name='cassandralin',
    version='0.0.1',
    packages=['cassandralin']
)

# These tools need to specify all of the necessary packages.

# numpy should be <= 1.23.5, otherwise GPy cannot be imported...
# then again, do we need to be able to import GPy? Let's see the
# minimum number of required packages we need in order to get
# this code working...
