# ElMD

![A drawing of ants moving earth](https://i.imgur.com/fg8Nrma.png)

The Element Movers Distance (ElMD) is a similarity measure for chemical compositions. This distance between two compositions is calculated from the minimal amount of work taken to transform one distribution of elements to another along the modified Pettifor scale. 

This repository provides the reference implementations as described in our paper "[The Earth Movers Distance as a metric for the space of inorganic compositions](https://chemrxiv.org/articles/preprint/The_Earth_Mover_s_Distance_as_a_Metric_for_the_Space_of_Inorganic_Compositions/12777566)". 

If you wish to compute this metric between lots of compositions, the ElM2D high-performance library may be more useful and can be found at www.github.com/lrcfmd/ElM2D.

We recommend installation via pip and python 3.7.

```
pip install python==3.7
pip install ElMD
```

## Usage
For simple usage initiate an object with its compositional formula

```python
> from ElMD import ElMD
> x = ElMD("CaTiO3")
```

Calculate the distance to a second object with the `elmd` method. 

```python
> x.elmd("SrTiO3")
0.2
```

Alternate chemical scales may be accessed via the "metric" argument, e.g.

```python
> x = ElMD("CaTiO3", metric="atomic")
> x.elmd("SrTiO3")
3.6
```

The `elmd()` method is overloaded to take two strings, with the choice of elemental metric taken from the first class.

```python
> elmd = ElMD().elmd
> elmd("NaCl", "LiCl")
0.5
```

## Elemental Similarity
You may use either traditional discrete scales or machine learnt representations for each element. In this instance a vector has been generated for each element, and the distance between elements (not compositions!) is the Euclidean distance. 

Due to the disparity in magnitudes of some of these values, a select few have additionally been scaled.

Linear:
- mendeleev
- petti
- atomic
- mod_petti

Chemically Derived:
- oliynyk 
- oliynyk_sc
- jarvis 
- jarvis_sc 
- magpie 
- magpie_sc 

Machine Learnt:
- cgcnn 
- elemnet 
- mat2vec 
- matscholar 
- megnet16 

Random Numbers:
- random_200

TODO HYPERLINK REFERENCES FOR DESCRIPTORS, MOSTLY FROM ROOST AND CRABNET.

The Euclidean distance between these vectors is taken as the measure of elemental similarity. 

```python
> x = ElMD("NaCl", metric="magpie")
> x.elmd("LiCl")
46.697806

> x = ElMD("NaCl", metric="magpie_sc")
> x.elmd("LiCl")
0.688539
```

The feature dictionary can be accessed through the `periodic_tab` attribute:

```python
> featurizingDict = ElMD().periodic_tab
> featurizingDict["magpie"]["Na"]
[2.0, 22.98976928, 370.87, 1.0, 3.0, 166.0, 0.93, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 29.2433333333, 0.0, 0.0, 229.0]
```

## Featurizing
Whilst not the initial purpose, a compositional based feature vector may be generated from ElMD objects should you require it. This is a mean pooling of the weighted composition feature matrix. 

Note that this vector representation is not used at any point during the ElMD distance calculation and is provided solely for convenience.

We construct this by taking the dot product of the ratios of each element with the features of these elements. Pass the argument feature_pooling="mean" to divide by the total number of elements in the compound.

```python
feature_vector = np.dot(ratio_vector, element_feature_matrix)
```

This is accessed through the `feature_vector` attribute.

```python
# For single element compositions, equivalent to x.periodic_tab["magpie"]["Cl"]
> x = ElMD("Cl", metric="magpie")
> x.feature_vector
array([ 94.    ,  35.453 , 171.6   ,  17.    ,   3.    , 102.    ,
         3.16  ,   2.    ,   5.    ,   0.    ,   0.    ,   7.    ,
         0.    ,   1.    ,   0.    ,   0.    ,   1.    ,  24.4975,
         2.493 ,   0.    ,  64.    ])

# Aggregate vector by each elements contribution
> x = ElMD("NaCl", metric="magpie").feature_vector
array([ 48.        ,  29.22138464, 271.235     ,   9.        ,
         3.        , 134.        ,   2.045     ,   1.5       ,
         2.5       ,   0.        ,   0.        ,   4.        ,
         0.5       ,   0.5       ,   0.        ,   0.        ,
         1.        ,  26.87041667,   1.2465    ,   0.        ,
       146.5       ])

# Divide this feature vector by the number of elements
> x = ElMD("NaCl", metric="magpie", feature_pooling="mean").feature_vector
array([ 24.        ,  14.61069232, 135.6175    ,   4.5       ,
         1.5       ,  67.        ,   1.0225    ,   0.75      ,
         1.25      ,   0.        ,   0.        ,   2.        ,
         0.25      ,   0.25      ,   0.        ,   0.        ,
         0.5       ,  13.43520833,   0.62325   ,   0.        ,
        73.25      ])
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
