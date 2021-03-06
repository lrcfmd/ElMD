# ElMD

The Element Movers Distance (ElMD) is a similarity measure for chemical compositions. This distance between two compositions is calculated from the minimal amount of work taken to transform one distribution of elements to another along the modified Pettifor scale. 

This repository provides the reference implementations as described in our paper "[The Earth Movers Distance as a metric for the space of inorganic compositions](https://chemrxiv.org/articles/preprint/The_Earth_Mover_s_Distance_as_a_Metric_for_the_Space_of_Inorganic_Compositions/12777566)". 

If you wish to compute this metric between lots of compositions, the ElM2D high-performance library may be more useful and can be found at [www.github.com/lrcfmd/ElMD](www.github.com/lrcfmd/ElMD).

We recommend installation via pip and python 3.7.

```
pip install python==3.7
pip install ElMD
```

## Usage
For simple usage initiate an object with its compositional formula

```python
from ElMD import ElMD
x = ElMD("CaTiO3")
```

Calculate the distance to a second object with the `elmd` method. 

```
> x.elmd("SrTiO3")
0.2
```

Alternate chemical scales may be accessed via the "metric" argument, e.g.

```python
> x = ElMD("CaTiO3", metric="atomic")
> x.elmd("SrTiO3")
3.6
```
You may use either traditional discrete scales or machine learnt representations for each element. In this instance a vector has been generated for each element, and the distance between elements is the Euclidean distance between these. Due to the disparity in magnitudes of these values, some of these have been scaled.

Linear:
- 'mendeleev'
- 'petti'
- 'atomic'
- 'mod_petti'

Machine Learnt:
- 'oliynyk' 
- 'olinyk_sc'
- 'cgcnn' 
- 'elemnet' 
- 'jarvis' 
- 'jarvis_sc' 
- 'magpie' 
- 'magpie_sc' 
- 'mat2vec' 
- 'matscholar' 
- 'megnet16' 
- 'random_200'

TODO HYPERLINK REFERENCES FOR DESCRIPTORS, MOSTLY FROM ROOST AND CRABNET.

The function is overloaded to take two strings (the metric argument is taken from the class) for ease with implementing.

```python
elmd = ElMD().elmd
elmd("NaCl", "LiCl")
```

## Featurizing
Whilst not the initial purpose, a feature vector may be generated from ElMD objects should you require it,. This is simply the dot product of the ratios of each element by the features of these elements, divided by the number of elements in the compound.

```python
> x = ElMD("NaCl", metric="magpie_sc")
> x.feature_vector
[-0.625      -0.08109291 -0.61183562 -0.57087692  0.0684659  -0.61122024
 -0.2043967   0.24933184 -0.37935793  0.37782711 -0.44429256 -0.33681995
 -0.40697298  0.45138889 -0.09132503 -0.31499791 -0.22061063 -0.36466139
  0.01337025  0.06100075 -0.10391629 -0.29680666]
```

## Documentation

Complete documentation may be found at www.elmd.io/api


## Citing

If you would like to cite this code in your work, please use the Chemistry of Materials reference

```
@article{doi:10.1021/acs.chemmater.0c03381,
author = {Hargreaves, Cameron J. and Dyer, Matthew S. and Gaultois, Michael W. and Kurlin, Vitaliy A. and Rosseinsky, Matthew J.},
title = {The Earth Mover’s Distance as a Metric for the Space of Inorganic Compositions},
journal = {Chemistry of Materials},
volume = {32},
number = {24},
pages = {10610-10620},
year = {2020},
doi = {10.1021/acs.chemmater.0c03381},
URL = { 
        https://doi.org/10.1021/acs.chemmater.0c03381
    
},
eprint = { 
        https://doi.org/10.1021/acs.chemmater.0c03381
    
}
}
```

## Issues

Please feel free to post (and reply to) any questions, comments, and legitimate concerns as issues on this GitHub page.