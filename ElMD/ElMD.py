'''
The Element Movers Distance is an application of the Wasserstein metric between
two compositional vectors

Copyright (C) 2020 Cameron Hargreaves
This file is part of The Element Movers Distance
<https://github.com/lrcfmd/ElMD>

The Element Movers Distance is free software: you can redistribute it and/or 
modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Element Movers Distance is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with dogtag.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Cameron Hargreaves"
__copyright__ = "2019, Cameron Hargreaves"
__credits__ = ["https://github.com/Zapaan", "Loïc Séguin-C. <loicseguin@gmail.com>", "https://github.com/Bowserinator/"]
__license__ = "GPL"
__version__ = "0.4.25"
__maintainer__ = "Cameron Hargreaves"

'''
import json 
import re
import os 
import pkg_resources

from site import getsitepackages
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import squareform
from numba import njit
from functools import lru_cache

#%%
# x = ElMD("LiF", metric="mod_petti")
# y = ElMD("NaCl", metric="mod_petti")

# print(x.elmd(y))
# print(y.elmd(x))

# z = x.full_feature_vector()
# print(z)

# #%%
# import matplotlib.pyplot as plt
# plt.pcolormesh(z, cmap='copper')


#%%
def main():
    import time 
    ts = time.perf_counter()
    x = ElMD("LiF", metric="mod_petti")
    y = ElMD("NaCl", metric="mod_petti")

    print(x.elmd(y))
    print(y.elmd(x))

    z = x.full_feature_vector()
    print(z)

    z = x.full_feature_vector(positional_encode=True)
    print(z)

    print(np.sum(np.abs(np.cumsum(x.ratio_vector - y.ratio_vector))))

    x = ElMD("Li7La3Hf2O12", metric="magpie")
    y = ElMD("CsPbI3", metric="magpie")
    # z = ElMD("Zr3AlN", metric="atomic")

    print(x.elmd(y))
    print(y.elmd(x))

    try:
        print(y.elmd(z))
    except Exception as e:
        print(e)
        z = ElMD("CaTiO5", metric="magpie")
        print(y.elmd(z))
    
    print(x)
    print(x.feature_vector)
    print(time.perf_counter() - ts)

@lru_cache(maxsize=16)
def _get_periodic_tab(metric):
    """
    Load periodic data from the python site_packages/ElMD folder
    """
    paths = getsitepackages()

    python_package_path = ""

    for p in paths:
        try:
            if "ElMD" in os.listdir(p):
                python_package_path = p
        except:
            pass 

    python_package_path = "" # For local debugging

    local_lookup_folder = os.path.join(python_package_path, "ElMD", "el_lookup")
    with open(os.path.join(local_lookup_folder, f"{metric}.json"), 'r') as j:
        ElementDict = json.loads(j.read())
            
    return ElementDict, local_lookup_folder
    

def elmd(comp1, comp2, metric="mod_petti"):
    if isinstance(comp1, str):
        comp1 = ElMD(comp1, metric=metric)
        source_demands = comp1.ratio_vector
    elif isinstance(comp1, ElMD):
        source_demands = comp1.ratio_vector
    else:
        raise TypeError(f"First composition must be either a string or ElMD object, you input an object of type {type(comp1)}")

    if isinstance(comp2, str):
        comp2 = ElMD(comp2, metric=metric)
        sink_demands = comp2.ratio_vector
    elif isinstance(comp2, ElMD):
        sink_demands = comp2.ratio_vector

    else:
        raise TypeError(f"Second composition must be either a string or ElMD object, you input an object of type {type(comp2)}")

    if isinstance(comp1, ElMD) and isinstance(comp2, ElMD) and comp1.metric != comp2.metric:
        raise TypeError(f"Both ElMD objects must use the same metric. comp1 has metric={comp1.metric} and comp2 has metric={comp2.metric}")

    source_labels = np.array([comp1.periodic_tab[comp1.lookup[i]] for i in np.where(source_demands > 0)[0]], dtype=float)
    sink_labels = np.array([comp2.periodic_tab[comp2.lookup[i]] for i in np.where(sink_demands > 0)[0]], dtype=float)
    
    source_demands = source_demands[np.where(source_demands > 0)[0]]
    sink_demands = sink_demands[np.where(sink_demands > 0)[0]]

    # Perform a floating point conversion to ints to ensure algorithm terminates
    network_costs = np.array([[np.linalg.norm(x - y) for x in sink_labels] for y in source_labels], dtype=np.float64) 

    return EMD(source_demands, sink_demands, network_costs)


def EMD(source_demands, sink_demands, network_costs):
    '''
    A numba compiled EMD function from the network simplex algorithm to compare 
    two distributions with a given distance matrix between node labels
    '''

    if len(network_costs.shape) == 2:
        n, m = network_costs.shape

        if len(source_demands) != n or len(sink_demands) != m:
            raise ValueError(f"Shape of 2D distance matrix must have n rows and m columns where n is the number of source_demands, and m is the number of sink demands. You have n={len(source_demands)} source_demands and m={len(sink_demands)} sink_demands, but your distance matrix is [{n}, {m}].")

        network_costs = network_costs.ravel()

    else:
        raise ValueError("Must input a 2D distance matrix between the elements of both distributions")

    return network_simplex(source_demands, sink_demands, network_costs)

class ElMD():
    ATOM_REGEX = r'([A-Z][a-z]*)(\d*\.?\d*[-+]?x?)'
    OPENERS = '({['
    CLOSERS = ')}]'

    # As the current optimization solver only takes in ints we must multiply
    # all floats to capture the decimal places
    FP_MULTIPLIER = 100000000

    def __init__(self, formula="", metric="mod_petti", strict_parsing=False, x=1):
        self.metric = metric
        
        if isinstance(formula, ElMD):
            return formula

        self.formula = formula.strip()
        self.strict_parsing = strict_parsing
        self.x = x
        
        self.periodic_tab, self.el_lookup_folder = _get_periodic_tab(metric)
        self.petti_lookup, _ = _get_periodic_tab("mod_petti")
        self.lookup = self._gen_lookup()
        # self.petti_lookup = self.filter_petti_lookup()

        self.composition = self._parse_formula(self.formula)
        self.normed_composition = self._normalise_composition(self.composition)
        self.ratio_vector = self._gen_ratio_vector()
        self.petti_vector = self._gen_petti_vector()

        self.pretty_formula = self._gen_pretty()

        self.feature_vector = self._gen_feature_vector()

    def full_feature_vector(self, positional_encode=False, min_freq=1e-4):
        feature_dicts = os.listdir(self.el_lookup_folder)
        
        # Skip unscaled chemical feature vectors
        feature_dicts = [f for f in feature_dicts if f[:-5] + "_sc.json" not in feature_dicts]
        
        # Strip redundant permutations of atomic number (included in other descriptors)
        feature_dicts = [f[:-5] for f in feature_dicts if f[:-5] not in ["atomic", "mendeleev", "petti", "mod_petti"]]

        if not positional_encode:
            full_features = []

            for f in feature_dicts:
                d, _ = _get_periodic_tab(f)
                vectors = np.array([d[el] for el in self.normed_composition.keys()])

                # mean of els
                full_features.append(np.mean(vectors, axis=0))
                
                weighted_features = np.array([r for r in self.normed_composition.values()]) * vectors.T
                weighted_features = weighted_features.T
                # weighted mean
                full_features.append(np.mean(weighted_features, axis=0))
                # min
                full_features.append(np.min(weighted_features, axis=0))
                # max
                full_features.append(np.max(weighted_features, axis=0))
                # range
                full_features.append(np.max(weighted_features, axis=0) - np.min(vectors, axis=0))
                # std
                full_features.append(np.std(weighted_features, axis=0))

            # convert to numpy array if single dimension (redundant?)
            full_features = [np.array([x]) if not isinstance(x, np.ndarray) else x for x in full_features]

            return np.concatenate(full_features)

        else:
            features = []

            for f in feature_dicts:
                d, _ = _get_periodic_tab(f)
                features.append(np.array([d[el] for el in self.normed_composition.keys()]))
            
            features = np.concatenate([x for x in features if x.ndim != 1], axis=1)

            d_model = features.shape[1] 

            # Fractional encoding as described by Anthony Wang: https://www.nature.com/articles/s41524-021-00545-1
            # Method taken from https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
            fraction = np.array([frac for frac in self.normed_composition.values()])
            freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)

            # Take a linear scaling of sin functions and a log
            # scaling of sin functions to capture small stoichiometries
            lin_pos_enc = fraction.reshape(-1,1) * freqs.reshape(1,-1)

            lin_pos_enc[:, ::2] = np.cos(lin_pos_enc[:, ::2])
            lin_pos_enc[:, 1::2] = np.sin(lin_pos_enc[:, 1::2])

            log_pos_enc = np.log10(fraction.reshape(-1,1) * freqs.reshape(1,-1))

            log_pos_enc[:, ::2] = np.cos(log_pos_enc[:, ::2])
            log_pos_enc[:, 1::2] = np.sin(log_pos_enc[:, 1::2])

            return features + lin_pos_enc + log_pos_enc

    def elmd(self, comp2 = None, comp1 = None):
        '''
        Calculate the minimal cost flow between two weighted vectors using the
        network simplex method. This is overloaded to accept a range of input
        types.
        '''
        if comp1 is None:
            comp1 = self

        if comp2 is None:
            raise TypeError("elmd() missing 1 required positional argument")

        return elmd(comp1, comp2, metric=self.metric)

    def _gen_ratio_vector(self):
        '''
        Create a numpy array from a composition dictionary. 
        '''
        comp = self.normed_composition
        
        if isinstance(comp, str):
            comp = self._parse_formula(comp)
            comp = self._normalise_composition(comp)

        comp_labels = []
        comp_ratios = []

        for k in sorted(comp.keys()):
            comp_labels.append(self.petti_lookup[k])
            comp_ratios.append(comp[k])

        indices = np.array(comp_labels, dtype=np.int64)
        ratios = np.array(comp_ratios, dtype=np.float64)

        numeric = np.zeros(shape=max([x for x in self.petti_lookup.values() if isinstance(x, int)]) + 1, dtype=np.float64)
        numeric[indices] = ratios

        return numeric

    def _gen_petti_vector(self):
        comp = self.normed_composition
        
        if isinstance(comp, str):
            comp = self._parse_formula(comp)
            comp = self._normalise_composition(comp)

        comp_labels = []
        comp_ratios = []

        for k in sorted(comp.keys()):
            comp_labels.append(self.petti_lookup[k])
            comp_ratios.append(comp[k])

        indices = np.array(comp_labels, dtype=np.int64)
        ratios = np.array(comp_ratios, dtype=np.float64)

        numeric = np.zeros(shape=len(self.petti_lookup), dtype=np.float64)
        numeric[indices] = ratios

        return numeric
        
    def _gen_feature_vector(self):
        """
        Perform the dot product between the ratio vector and its elemental representation. 
        """
        # If we only have an integer representation, return the vector as is
        if type(self.periodic_tab["H"]) is int:
            return self.ratio_vector
        
        n = int(len(self.normed_composition))
        m = len(self.periodic_tab["H"])
        numeric = np.zeros(shape=(n, m), dtype=float)

        els = list(self.periodic_tab.keys())

        for i, k in enumerate(self.normed_composition.keys()):
            try:
                numeric[i] = self.periodic_tab[k]
            except:
                print(f"Failed to process {self.formula} with {self.metric} due to element {k}, discarding this element.")

        element_features = np.nan_to_num(numeric)

        weighted_vector = np.dot(self.ratio_vector[np.where(self.ratio_vector > 0)[0]], element_features)

        return weighted_vector

    def _gen_pretty(self):
        '''
        Return a normalized formula string ordered by the mod_petti dictionary
        '''
        inds = np.where(self.petti_vector != 0.0)[0]
        pretty_form = ""

        for i, ind in enumerate(inds):
            if self.petti_vector[ind] == 1:
                pretty_form = pretty_form + f"{self.lookup[ind]}"
            else:
                pretty_form = pretty_form + f"{self.lookup[ind]}{self.petti_vector[ind]:.3f}".strip('0') + ' '

        return pretty_form.strip()

    def _gen_lookup(self):
        lookup = {}
        
        for i, (k, v) in enumerate(self.petti_lookup.items()):
            lookup[k] = v 
            lookup[int(v)] = k 

        return lookup

    def _is_balanced(self, formula):
        """Check if all sort of brackets come in pairs."""
        c = Counter(formula)
        return c['['] == c[']'] and c['{'] == c['}'] and c['('] == c[')']

    def _dictify(self, tuples):
        """Transform tuples of tuples to a dict of atoms."""
        res = dict()

        for atom, n in tuples:
            if atom[-1].lower() == "x":
                if self.strict_parsing:
                    raise ValueError(f"The element {atom} in the composition {self.formula} is undefined. Set strict_parsing=False to read x=1.")
                else:
                    atom = atom[:-1]
                    n = self.x
            
            if not isinstance(n, (int, float)) and "x" in n and "-" in n:
                n = float(str.split(n, "-")[0]) - self.x 
                if n < 0: n = 0

            if not isinstance(n, (int, float)) and "x" in n and "+" in n:
                n = float(str.split(n, "+")[0]) + self.x 
            
            try:
                if atom in self.lookup:
                    res[atom] += float(n or 1)
                elif self.strict_parsing:
                    raise ValueError(f"The element {atom} in the composition {self.formula} is not in the lookup dictionary for {self.metric}. Set strict_parsing=False to skip this element")

            except KeyError:
                res[atom] = float(n or 1)

            except ValueError as e:
                raise e

        return res

    def _fuse(self, mol1, mol2, w=1):
        """ Fuse 2 dicts representing molecules. Return a new dict. """
        return {atom: (mol1.get(atom, 0) + mol2.get(atom, 0)) * w for atom in set(mol1) | set(mol2)}

    def _parse(self, formula):
        """
        Return the molecule dict and length of parsed part.
        Recurse on opening brackets to parse the subpart and
        return on closing ones because it is the end of said subpart.
        """
        q = []
        mol = {}
        i = 0

        while i < len(formula):
            # Using a classic loop allow for manipulating the cursor
            token = formula[i]

            if token in self.CLOSERS:
                # Check for an index for this part
                m = re.match(r'\d+\.*\d*|\.\d*', formula[i+1:])
                if m:
                    weight = float(m.group(0))
                    i += len(m.group(0))
                else:
                    weight = 1

                submol = self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))
                return self._fuse(mol, submol, weight), i

            elif token in self.OPENERS:
                submol, l = self._parse(formula[i+1:])
                mol = self._fuse(mol, submol)
                # skip the already read submol
                i += l + 1
            else:
                q.append(token)

            i += 1

        # Fuse in all that's left at base level
        return self._fuse(mol, self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))), i

    def _parse_formula(self, formula):
        """Parse the formula and return a dict with occurences of each atom."""
        if not self._is_balanced(formula):
            raise ValueError("Your brackets not matching in pairs ![{]$[&?)]}!]")

        return self._parse(formula)[0]

    def _normalise_composition(self, input_comp):
        """ Sum up the numbers in our counter to get total atom count """
        composition = deepcopy(input_comp)
        # check it has been processed
        if isinstance(composition, str):
            composition = self._parse_formula(composition)

        atom_count =  sum(composition.values(), 0.0)

        for atom in composition:
            composition[atom] /= atom_count

        return composition

    def _get_atomic_num(self, element):
        """ Return atomic number from element """
        try:
            np.array(self.periodic_tab[element])
        except Exception as e:
            if self.strict_parsing:
                raise Exception(f"Element, {element} not found in lookup dict {self.metric}, in composition {self.formula}")
            else:
                return 0

    def _get_position(self, element):
        """
        Return either the x, y coordinate of an elements position, or the
        x-coordinate on the Pettifor numbering system as a 2-dimensional
        """
        keys = list(self.periodic_tab.keys())

        try:
            atomic_num = keys.index(element)
            return atomic_num

        except:
            if self.strict_parsing:
                raise KeyError(f"One of the elements in {self.composition} is not in the {self.metric} dictionary. Try a different representation or use strict_parsing=False")
            else:
                return -1

    def _return_positions(self, composition):
        """ Return a dictionary of associated positions for each element """
        element_pos = {}

        for element in composition:
            element_pos[element] = self._get_position(element)

        return element_pos

    def __repr__(self):
        return f"ElMD({self.pretty_formula})"

    def __len__(self):
        return len(self.normed_composition)

    def __eq__(self, other):
        return self.pretty_formula == other.pretty_formula

    def __ne__(self, other):
        return self.pretty_formula != other.pretty_formula
    
    def __lt__(self, other):
        return self.elmd("H") < other.elmd("H")

    def __gt__(self, other):
        return self.elmd("H") > other.elmd("H")


'''
This is an implementation of the network simplex algorithm for computing the
minimal flow atomic similarity distance between two compounds

Copyright (C) 2019  Cameron Hargreaves
ported from networkx to numba/numpy, Copyright (C) 2010 Loïc Séguin-C.
All rights reserved.
BSD license.
'''

@njit(cache=True)
def reduced_cost(i, costs, potentials, tails, heads, flows):
    """Return the reduced cost of an edge i.
    """
    c = costs[i] - potentials[tails[i]] + potentials[heads[i]]

    if flows[i] == 0:
        return c
    else:
        return -c

@njit(cache=True)
def find_entering_edges(B, e, f, tails, heads, costs, potentials, flows):
    """Yield entering edges until none can be found.
    """
    # Entering edges are found by combining Dantzig's rule and Bland's
    # rule. The edges are cyclically grouped into blocks of size B. Within
    # each block, Dantzig's rule is applied to find an entering edge. The
    # blocks to search is determined following Bland's rule.

    M = (e + B - 1) // B    # number of blocks needed to cover all edges
    m = 0

    while m < M:
        # Determine the next block of edges.
        l = f + B
        if l <= e:
            edge_inds = np.arange(f, l)
        else:
            l -= e
            edge_inds = np.empty(e - f + l, dtype=np.int64)
            for i, v in enumerate(range(f, e)):
                edge_inds[i] = v
            for i in range(l):
                edge_inds[e - f + i] = i

        f = l

        # Find the first edge with the lowest reduced cost.
        i = edge_inds[0]
        c = reduced_cost(i, costs, potentials, tails, heads, flows)
        for ind in edge_inds[1:]:
            cost = reduced_cost(ind, costs, potentials, tails, heads, flows)
            if cost < c:
                c = cost
                i = ind

        p = q = -1

        if c >= 0:
            m += 1

        # Entering edge found.
        else:
            if flows[i] == 0:
                p = tails[i]
                q = heads[i]
            else:
                p = heads[i]
                q = tails[i]

            return i, p, q, f

    # All edges have nonnegative reduced costs. The flow is optimal.
    return -1, -1, -1, -1

@njit(cache=True)
def find_apex(p, q, size, parent):
    """Find the lowest common ancestor of nodes p and q in the spanning
    tree.
    """
    size_p = size[p]
    size_q = size[q]

    while True:
        while size_p < size_q:
            p = parent[p]
            size_p = size[p]
        while size_p > size_q:
            q = parent[q]
            size_q = size[q]
        if size_p == size_q:
            if p != q:
                p = parent[p]
                size_p = size[p]
                q = parent[q]
                size_q = size[q]
            else:
                return p
@njit(cache=True)
def trace_path(p, w, edge, parent):
    """Return the nodes and edges on the path from node p to its ancestor
    w.
    """
    cycle_nodes = [p]
    cycle_edges = []

    while p != w:
        cycle_edges.append(edge[p])
        p = parent[p]
        cycle_nodes.append(p)

    return cycle_nodes, cycle_edges

@njit(cache=True)
def find_cycle(i, p, q, size, edge, parent):
    """Return the nodes and edges on the cycle containing edge i == (p, q)
    when the latter is added to the spanning tree.

    The cycle is oriented in the direction from p to q.
    """
    w = find_apex(p, q, size, parent)
    cycle_nodes, cycle_edges = trace_path(p, w, edge, parent)
    cycle_nodes_rev, cycle_edges_rev = trace_path(q, w, edge, parent)
    append_i_to_edges = (len(cycle_edges) < 1 or cycle_edges[-1] != i)
    add_to_c_nodes = max(len(cycle_nodes_rev) - 1, 0)
    cycle_nodes_ = np.empty(len(cycle_nodes) + add_to_c_nodes, dtype=np.int64)

    for j in range(len(cycle_nodes)):
        cycle_nodes_[j] = cycle_nodes[-(j+1)]
    for j in range(add_to_c_nodes):
        cycle_nodes_[len(cycle_nodes) + j] = cycle_nodes_rev[j]

    if append_i_to_edges:
        cycle_edges_ = np.empty(len(cycle_edges) + len(cycle_edges_rev) + 1, dtype=np.int64)
        cycle_edges_[len(cycle_edges)] = i
    else:
        cycle_edges_ = np.empty(len(cycle_edges) + len(cycle_edges_rev), dtype=np.int64)

    for j in range(len(cycle_edges)):
        cycle_edges_[j] = cycle_edges[-(j+1)]
    for j in range(1, len(cycle_edges_rev) + 1):
        cycle_edges_[-j] = cycle_edges_rev[-j]

    return cycle_nodes_, cycle_edges_

@njit(cache=True)
def residual_capacity(i, p, capac, flows, tails):
    """Return the residual capacity of an edge i in the direction away
    from its endpoint p.
    """
    if tails[i] == p:
        return capac[i] - flows[i]
    else:
        return flows[i]


@njit(cache=True)
def find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads):
    """Return the leaving edge in a cycle represented by cycle_nodes and
    cycle_edges.
    """
    j, s = cycle_edges[0], cycle_nodes[0]
    res_caps_min = residual_capacity(j, s, capac, flows, tails)
    for ind in range(1, cycle_edges.shape[0]):
        j_, s_ = cycle_edges[ind], cycle_nodes[ind]
        res_cap = residual_capacity(j_, s_, capac, flows, tails)
        if res_cap < res_caps_min:
            res_caps_min = res_cap
            j, s = j_, s_

    t = heads[j] if tails[j] == s else tails[j]
    return j, s, t

@njit(cache=True)
def augment_flow(cycle_nodes, cycle_edges, f, tails, flows):
    """Augment f units of flow along a cycle representing Wn with cycle_edges.
    """
    for i, p in zip(cycle_edges, cycle_nodes):
        if tails[i] == p:
            flows[i] += f
        else:
            flows[i] -= f

@njit(cache=True)
def trace_subtree(p, last, next):
    """Yield the nodes in the subtree rooted at a node p.
    """
    tree = []
    tree.append(p)

    l = last[p]
    while p != l:
        p = next[p]
        tree.append(p)

    return np.array(tree, dtype=np.int64)

@njit(cache=True)
def remove_edge(s, t, size, prev, last, next, parent, edge):
    """Remove an edge (s, t) where parent[t] == s from the spanning tree.
    """
    size_t = size[t]
    prev_t = prev[t]
    last_t = last[t]
    next_last_t = next[last_t]
    # Remove (s, t).
    parent[t] = -2
    edge[t] = -2
    # Remove the subtree rooted at t from the depth-first thread.
    next[prev_t] = next_last_t
    prev[next_last_t] = prev_t
    next[last_t] = t
    prev[t] = last_t

    # Update the subtree sizes and last descendants of the (old) ancestors
    # of t.
    while s != -2:
        size[s] -= size_t
        if last[s] == last_t:
            last[s] = prev_t
        s = parent[s]

@njit(cache=True)
def make_root(q, parent, size, last, prev, next, edge):
    """Make a node q the root of its containing subtree.
    """
    ancestors = []
    # -2 means node is checked
    while q != -2:
        ancestors.insert(0, q)
        q = parent[q]

    for p, q in zip(ancestors[:-1], ancestors[1:]):
        size_p = size[p]
        last_p = last[p]
        prev_q = prev[q]
        last_q = last[q]
        next_last_q = next[last_q]

        # Make p a child of q.
        parent[p] = q
        parent[q] = -2
        edge[p] = edge[q]
        edge[q] = -2
        size[p] = size_p - size[q]
        size[q] = size_p

        # Remove the subtree rooted at q from the depth-first thread.
        next[prev_q] = next_last_q
        prev[next_last_q] = prev_q
        next[last_q] = q
        prev[q] = last_q

        if last_p == last_q:
            last[p] = prev_q
            last_p = prev_q

        # Add the remaining parts of the subtree rooted at p as a subtree
        # of q in the depth-first thread.
        prev[p] = last_q
        next[last_q] = p
        next[last_p] = q
        prev[q] = last_p
        last[q] = last_p

@njit(cache=True)
def add_edge(i, p, q, next, prev, last, size, parent, edge):
    """Add an edge (p, q) to the spanning tree where q is the root of a
    subtree.
    """
    last_p = last[p]
    next_last_p = next[last_p]
    size_q = size[q]
    last_q = last[q]
    # Make q a child of p.
    parent[q] = p
    edge[q] = i
    # Insert the subtree rooted at q into the depth-first thread.
    next[last_p] = q
    prev[q] = last_p
    prev[next_last_p] = last_q
    next[last_q] = next_last_p

    # Update the subtree sizes and last descendants of the (new) ancestors
    # of q.
    while p != -2:
        size[p] += size_q
        if last[p] == last_p:
            last[p] = last_q
        p = parent[p]

@njit(cache=True)
def update_potentials(i, p, q, heads, potentials, costs, last, next):
    """Update the potentials of the nodes in the subtree rooted at a node
    q connected to its parent p by an edge i.
    """
    if q == heads[i]:
        d = potentials[p] - costs[i] - potentials[q]
    else:
        d = potentials[p] + costs[i] - potentials[q]

    potentials[q] += d
    l = last[q]
    while q != l:
        q = next[q]
        potentials[q] += d

@njit(cache=True)
def network_simplex(source_demands, sink_demands, network_costs):
    '''
    This is a port of the network simplex algorithm implented by Loïc Séguin-C
    for the networkx package to allow acceleration via the numba package

    Copyright (C) 2010 Loïc Séguin-C. <loicseguin@gmail.com>
    All rights reserved.
    BSD license.

    References
    ----------
    .. [1] Z. Kiraly, P. Kovacs.
           Efficient implementation of minimum-cost flow algorithms.
           Acta Universitatis Sapientiae, Informatica 4(1):67--118. 2012.
    .. [2] R. Barr, F. Glover, D. Klingman.
           Enhancement of spanning tree labeling procedures for network
           optimization.
           INFOR 17(1):16--34. 1979.
    '''
    n_sources, n_sinks = source_demands.shape[0], sink_demands.shape[0]
    network_costs = network_costs.ravel()
    n = n_sources + n_sinks
    e = n_sources * n_sinks
    B = np.int64(np.ceil(np.sqrt(e)))
    
    # Constant used throughout for conversions from floating point to integer
    fp_multiplier = np.int64(1_000_000)

    # Multiply by a large number and cast to int to remove floating points
    # Add one additional node for a dummy source and sink
    source_d_int = (source_demands * fp_multiplier).astype(np.int64)
    sink_d_int = (sink_demands * fp_multiplier).astype(np.int64)

    # FP conversion error correction
    source_sum = np.sum(source_d_int)
    sink_sum = np.sum(sink_d_int)
    sink_source_sum_diff = sink_sum - source_sum

    if sink_source_sum_diff > 0:
        source_d_int[np.argmax(source_d_int)] += sink_source_sum_diff
    elif sink_source_sum_diff < 0:
        sink_d_int[np.argmax(sink_d_int)] -= sink_source_sum_diff

    # Create demands array
    demands = np.empty(n, dtype=np.int64)
    demands[:n_sources] = -source_d_int
    demands[n_sources:] = sink_d_int

    # Create fully connected arcs between all sources and sinks
    tails = np.empty(e + n, dtype=np.int64)
    heads = np.empty(e + n, dtype=np.int64)

    for i in range(n_sources):
        for j in range(n_sinks):
            ind = i * n_sinks + j
            tails[ind] = i
            heads[ind] = j + n_sources
    
    for i, demand in enumerate(demands):
        if demand > 0:
            tails[e + i] = -1
            heads[e + i] = -1
        else:
            tails[e + i] = i
            heads[e + i] = i

    # Create costs and capacities for the arcs between nodes
    network_costs = network_costs * fp_multiplier

    network_capac = np.array([min(source_demands[i], sink_demands[j])
                              for i in range(n_sources)
                              for j in range(n_sinks)],
                             dtype=np.float64) * fp_multiplier

    # Set a suitably high integer for infinity
    faux_inf = 3 * np.max(np.array((
                            np.sum(network_capac),
                            np.sum(np.abs(network_costs)),
                            np.amax(source_d_int),
                            np.amax(sink_d_int)),
                          dtype=np.int64))

    # allocate arrays
    costs = np.empty(e + n, dtype=np.int64)
    costs[:e] = network_costs
    costs[e:] = faux_inf

    capac = np.empty(e + n, dtype=np.int64)
    capac[:e] = network_capac
    capac[e:] = fp_multiplier

    flows = np.empty(e + n, dtype=np.int64)
    flows[:e] = 0
    flows[e:e+n_sources] = source_d_int
    flows[e+n_sources:] = sink_d_int

    potentials = np.empty(n, dtype=np.int64)
    demands_neg_mask = demands <= 0
    potentials[demands_neg_mask] = faux_inf
    potentials[~demands_neg_mask] = -faux_inf

    parent = np.empty(n + 1, dtype=np.int64)
    parent[:-1] = -1
    parent[-1] = -2

    size = np.empty(n + 1, dtype=np.int64)
    size[:-1] = 1
    size[-1] = n + 1

    next_node = np.arange(1, n + 2, dtype=np.int64)
    next_node[-2] = -1
    next_node[-1] = 0

    last_node = np.arange(n + 1, dtype=np.int64)
    last_node[-1] = n - 1

    prev_node = np.arange(-1, n, dtype=np.int64)
    edge = np.arange(e, e + n, dtype=np.int64)

    ###########################################################################
    # Main Pivot loop
    ###########################################################################

    f = 0
    while True:
        i, p, q, f = find_entering_edges(B, e, f, tails, heads, costs, potentials, flows)
        if p == -1: # If no entering edges then the optimal score is found
            break

        cycle_nodes, cycle_edges = find_cycle(i, p, q, size, edge, parent)
        j, s, t = find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads)
        augment_flow(cycle_nodes, cycle_edges, residual_capacity(j, s, capac, flows, tails), tails, flows)

        if i != j:  # Do nothing more if the entering edge is the same as the leaving edge.
            if parent[t] != s:
                # Ensure that s is the parent of t.
                s, t = t, s

            # Ensure that q is in the subtree rooted at t.
            for val in cycle_edges:
                if val == j:
                    p, q = q, p
                    break
                elif val == i:
                    break

            remove_edge(s, t, size, prev_node, last_node, next_node, parent, edge)
            make_root(q, parent, size, last_node, prev_node, next_node, edge)
            add_edge(i, p, q, next_node, prev_node, last_node, size, parent, edge)
            update_potentials(i, p, q, heads, potentials, costs, last_node, next_node)

    final_flows = flows[:e] / fp_multiplier
    edge_costs = costs[:e] / fp_multiplier
    final = final_flows.dot(edge_costs)

    return final

if __name__ == "__main__":
    main()
