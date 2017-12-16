from setuptools import setup


setup(name='BigProject',
      version='0.1',
      description='TBF',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
      author='Pierre Springer',
      author_email='pierre.springer@tum.de',
      license='MIT',
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['funniest-joke=funniest.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
      
