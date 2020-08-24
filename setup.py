from distutils.core import setup
setup(
  name = 'ElMD',        
  packages = ['ElMD'],  
  version = '0.1',      
  license='GPL3',       
  description = 'An implementation of the Element movers distance for chemical similarity of ionic compositions',  
  author = 'Cameron Hagreaves',              
  author_email = 'cameron.h@rgreaves.me.uk', 
  url = 'https://github.com/lrcfmd/ElMD/',   
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    
  keywords = ['ChemInformatics', 'Materials Science', 'Machine Learning', 'Materials Representation'],   
  install_requires=[            
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',  
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: GNU General Public License v3.0',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)