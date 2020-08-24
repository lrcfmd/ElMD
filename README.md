# ElMD
The Element Movers Distance (ElMD) is a similarity measure for chemical compositions. This distance between two compositions is calculated from the minimal amount of work taken to transform one distribution of elements to another along the modified Pettifor scale. 

This repository provides the reference implementations as described in our paper "[The Earth Movers Distance as a metric for the space of inorganic compositions](https://chemrxiv.org/articles/preprint/The_Earth_Mover_s_Distance_as_a_Metric_for_the_Space_of_Inorganic_Compositions/12777566)". 

We recommend installation via pip

`pip install ElMD`

## Usage
`from ElMD import ElMD`

`x = ElMD("CaTiO3")`

`x.elmd("SrTiO3")` 

`0.2`


