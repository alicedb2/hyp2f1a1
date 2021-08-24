from setuptools import setup
from Cython.Build import cythonize


setup(name='hyp2f1a1',
      version='0.1',
      description='Fast continued-fraction implementation of the Gauss hypergeometric function 2F1 for the special case alpha=1',
      url='http://github.com/alicedb2/hyp2f1a1',
      author='Alice Doucet Beaupre',
      author_email='alice.doucet.beaupre@gmail.com',
      license='MIT',
      packages=['hyp2f1a1'],
      ext_modules=cythonize('hyp2f1a1/hyp2f1a1.pyx'),
      zip_safe=False)
