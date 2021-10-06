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
> elmd = ElMD(metric="magpie_sc").elmd
> elmd("NaCl", "LiCl")
0.688539
```

The compositional parser can handle user defined values of `x` when this is applicable.

```python
latp_02 = ElMD("Li1+xAlxTi2-x(PO4)3", x=0.2) # Li1.2Al0.2Ti1.8(PO4)3
latp_03 = ElMD("Li1+xAlxTi2-x(PO4)3", x=0.3) # Li1.3Al0.3Ti1.7(PO4)3
```

## Elemental Similarity
You may use either traditional discrete scales or machine learnt representations for each element. In this instance a vector has been generated for each element, and the distance between elements (not compositions!) is the Euclidean distance. 

Due to the disparity in magnitudes of some of these values, a select few have additionally been scaled.

Linear:
- [mendeleev](https://www.sciencedirect.com/science/article/abs/pii/S0925838803008004)
- [petti](https://www.sciencedirect.com/science/article/abs/pii/S0925838803008004)
- [atomic](https://www.sciencedirect.com/science/article/abs/pii/S0925838803008004)
- [mod_petti](https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011/meta)

Chemically Derived:
- [oliynyk](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [oliynyk_sc](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [jarvis](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [jarvis_sc](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [magpie](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [magpie_sc](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)

Machine Learnt:
- [cgcnn](https://github.com/CompRhys/roost/tree/master/data/embeddings)
- [elemnet](https://github.com/CompRhys/roost/tree/master/data/embeddings)
- [mat2vec](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)
- [matscholar](https://github.com/CompRhys/roost/tree/master/data/embeddings)
- [megnet16](https://github.com/CompRhys/roost/tree/master/data/embeddings)

Random Numbers:
- [random_200](https://github.com/anthony-wang/CrabNet/tree/master/data/element_properties)

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
> featurizingDict = ElMD(metric="magpie).periodic_tab
> featurizingDict["Na"]
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
# For single element compositions, equivalent to x.periodic_tab["Cl"]
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

Please feel free to post any questions or comments as issues on this GitHub page.
