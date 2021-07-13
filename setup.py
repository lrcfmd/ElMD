from setuptools import setup, find_packages

setup(
  name = 'ElMD',        
  packages = ['ElMD'],  
  version = '0.3.11',      
  license='GPL3',       
  description = 'An implementation of the Element movers distance for chemical similarity of ionic compositions',  
  author = 'Cameron Hagreaves',            
  author_email = 'cameron.h@rgreaves.me.uk', 
  url = 'https://github.com/lrcfmd/ElMD/',   
  download_url = 'https://github.com/lrcfmd/ElMD/archive/v0.3.9.tar.gz',    
  keywords = ['ChemInformatics', 'Materials Science', 'Machine Learning', 'Materials Representation'],  
  package_data={"elementFeatures": ["ElementDict.json"]}, 
  include_package_data=True,
  install_requires=[ 
          'setuptools',
          'numba',
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',  
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3) ',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)