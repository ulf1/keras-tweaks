from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='keras-tweaks',
      version='0.1.0',
      description='lorem ipsum',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/ulf1/keras-tweaks',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['keras-tweaks'],
      install_requires=[
          'setuptools>=40.0.0',
          'tensorflow>=2.2.0'],
      python_requires='>=3.8',
      zip_safe=False)
