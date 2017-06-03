#!/usr/bin/python3.5
"""
Arthur Zwaenepoel
"""

from numpy import inf
from heapq import heappop, heappush, heappushpop, heapify
import numpy as np
import pandas as pd
import os
import re
import itertools
from ete3 import Tree
from io import StringIO


def proportion_different_sites(s1, s2):
    """
    Calculate the proportion of different sites between pairwise sequences
    :param s1:
    :param s2:
    :return:
    """
    p = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            p += 1
    return p / len(s1)


def jukes_cantor(s1, s2):
    """
    Implementation of the JC69 model (Jukes & Cantor, 1969) which assumes
    that every nucleotide has the same instantaneous rate of changing into another
    nucleotide (an assumption that is almost never valid).
    :return:
    """
    p = proportion_different_sites(s1, s2)
    d = -(3 / 4) * np.log(1 - (4 / 3) * p)
    var_d = (p * (1 - p) / len(s1)) * (1 / (1 - 4 * p / 3))
    return d, var_d


def distance_matrix(msa, distance='JC69'):
    """
    Function to get a distance matrix from a multiple sequence alignment
    :param msa: dictionary with aligned sequences
    :param distance: distance metric to use (currently only JC69 supported)
    :return: distance matrix (pandas data frame)
    """
    l = list(msa.keys())
    df = pd.DataFrame(np.zeros((len(l), len(l))), index=l, columns=l)

    for i in range(len(list(df.index))):
        gene_1 = df.index[i]

        for j in range(i + 1, len(list(df.index))):
            gene_2 = df.index[j]

            if distance == 'JC69':
                d = jukes_cantor(msa[gene_1], msa[gene_2])
                df[gene_1][gene_2] = d[0]
                df[gene_2][gene_1] = d[0]

    return df


def read_msa(msa):
    """
    Read multiple sequence alignment (MSA) in PHYLIP format.
    :param msa: MSA file
    :return: dictionary
    """
    msa_dict = {}

    with open(msa, 'r') as f:
        content = f.readlines()

    n_seq, length = int(content[0].split()[0].strip()), int(content[0].split()[1].strip())

    for i in range(1, n_seq * 2, 2):
        msa_dict[content[i].strip()] = content[i + 1].strip()

    return msa_dict


def get_species(gene, species):
    """
    Get species for a gene using regex matches.
    """
    for sp, p in species.items():
        if p.match(gene):
            return sp
    raise ValueError("species not found for gene {}!".format(gene))


def collapse_on_species(distance_matrix, species):
    """
    Collapse a given distance matrix for a gene family on species level.
    Distances for duplicates within a species are averaged.
    """
    matrix = pd.DataFrame()
    sp_set = set()
    to_drop = []
    for gene in distance_matrix.index:
        sp = get_species(gene, species)
        if sp in sp_set:
            matrix[sp] += distance_matrix[gene]
            matrix[sp] /= 2
            to_drop.append(gene)
        else:
            matrix[sp] = distance_matrix[gene]
            sp_set.add(sp)

    matrix = matrix.drop(to_drop)
    matrix.index = matrix.columns

    for k in range(len(matrix.index)):
        matrix.iloc[k, k] = 0
    return matrix


def topology_matrix_dijkstra(tree, species):
    """
    Input: newick tree
    Output: topology matrix
    """
    t = Tree(tree)
    i = len(species)
    for node in t.traverse():
        if node.name in species:
            node.name = species[node.name]
        else:
            node.name = i
            i += 1

    graph = tree_to_adj_list(t)
    leaves = list(t.get_leaves())
    paths = {}
    branches = set()
    for i in range(len(leaves)):
        node1 = leaves[i]
        for j in range(i + 1, len(leaves)):
            node2 = leaves[j]
            path, distance = dijkstra(graph, node1.name, node2.name)
            path = [(path[x], path[x + 1]) for x in range(len(path) - 1)]
            paths[tuple(sorted([node1.name, node2.name]))] = path
            for p in path:
                p = tuple(sorted(p))
                if p not in branches:
                    branches.add(p)

    # put in a matrix
    df = pd.DataFrame(index=list(paths.keys()), columns=list(branches))
    for tup, path in paths.items():
        for branch in path:
            branch = tuple(sorted(branch))
            df[branch][tup] = 1
    df = df.fillna(0)
    df.sort_index(0, inplace=True)
    df.sort_index(1, inplace=True)
    return df, t


def dijkstra(graph, source, sink=None):
    """
    Implementation of Dijkstra's shortest path algorithm
    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node
        - sink : the sink node (optional)
    Ouput:
        - distance : dict with the distances of the nodes to the source
        - came_from : dict with for each node the came_from node in the shortest
                    path from the source
    """
    distance = {v: inf for v in graph}
    distance[source] = 0
    current = source
    previous = {}
    Q = [(source, 0)]

    while Q:
        U = heappop(Q)
        if sink and U == sink:
            break
        for V in graph[U[0]]:
            alt = distance[U[0]] + V[0]
            if alt < distance[V[1]]:
                distance[V[1]] = alt
                previous[V[1]] = U[0]
                heappush(Q, (V[1], alt))

    if sink is None:
        return distance, previous
    else:
        return reconstruct_path(previous, source, sink), distance[sink]


def reconstruct_path(previous, source, sink):
    """
    Reconstruct the path from the output of the Dijkstra algorithm
    Inputs:
            - previous : a dict with the came_from node in the path
            - source : the source node
            - sink : the sink node
    Ouput:
            - the shortest path from source to sink (list)
    """
    if sink not in previous:
        return []

    V = sink
    path = [V]
    while V != source:
        V = previous[V]
        path.append(V)
    return path


def tree_to_adj_list(tree):
    """
    Convert ete3 Tree object to an adjacency list representing the tree graph
    """
    adj_list = {}
    for node in tree.traverse('postorder'):
        l = []
        if not node.is_leaf():
            # add children
            l += [(1, c.name) for c in node.children]
        if not node.is_root():
            # add parent
            l.append((1, node.up.name))
        adj_list[node.name] = set(l)
    return adj_list


def code_species(species):
    """
    Code species to integers
    """
    if type(species) == dict:
        species = list(species.keys())
    species_to_int = {x: 0 for x in species}
    i = 0
    for x in sorted(species):
        species_to_int[x] = i
        i += 1
    return species_to_int


def sub_topology_matrix(subset, matrix):
    """
    Get topology matrix for a subset of taxa
    """
    to_drop = []
    for pair in matrix.index:
        if pair[0] not in subset or pair[1] not in subset:
            to_drop.append(pair)
    matrix = matrix.drop(to_drop)
    return matrix


def sequence_length(msa_file):
    """
    Construct a weight matrix. Using alignment length as weight.
    """
    with open(msa_file, 'r') as f:
        content = f.readlines()

    length = int(content[0].split()[1].strip())
    return length


def weight_matrix(length, s):
    """
    Construct a weight matrix. Using alignment length as weight.
    """
    return np.diag([length for i in range(s)])


def vectorize_delta(delta):
    """
    Convert distance matrix to vector
    """
    d = np.array(delta)
    l = []
    for i in range(0, len(d)):
        for j in range(i + 1, len(d)):
            l.append(d[i, j])
    return np.array(l)


def solve_naively(D, B, C, z, Z):
    naive = np.bmat(
        [[D, B.T, z],
         [B, C, np.ones((B.shape[0], 1))],
         [z.T, np.zeros((1, C.shape[1])), np.array(0).reshape(1, 1)]]
    )
    right = np.zeros((1, naive.shape[0]))
    right[:, -1] = Z
    right = right.reshape(-1, 1)
    solution = np.linalg.solve(naive, right)

    return solution[D.shape[0]:-1], solution[:D.shape[0]]


def solve_cleverly(D, B, C, z, Z):
    # invert D, diagonal matrix so inverse is just diagonal elements^-1
    #D_inv = np.linalg.inv(D)
    D_inv = 1/D
    D_inv[np.isinf(D_inv)] = 0

    u = B @ D_inv @ z
    w = (z.T @ D_inv @ z)[0][0]
    M = C + ((1 / w) * u @ u.T) - B @ D_inv @ B.T
    # print(M)
    b = np.linalg.solve(M, -(Z / w) * u)
    alpha = D_inv @ ((z @ u.T / w) - B.T) @ b + (Z / w) * D_inv @ z
    # alpha = D_inv @ ((z.reshape((1,-1)) @ u.reshape((-1,1)))/w - B.T) @ b + (Z/w)*D_inv @ z
    # alpha[alpha == 0] = 0.0001 # this shouldn't be necessary

    return b, alpha


def parse_distance_matrices(distances_file):
    """
    Parse ditances as provided by ERaBLE authors

    :param distances_file: file with distance matrices
    :return: distances data frames in list and sequence lengths
    """
    lengths = []
    matrices = []

    with open(distances_file, 'r') as f:
        content = f.read()

    content = content.split('\n\n')
    for mat in content[1:]:
        mat = mat.split('\n')
        i = 0
        for i in range(len(mat)):
            if mat[i] != '':
                break
        if len(mat[i].split()) == 2:
            lengths.append(int(mat[i].strip().split()[1]))
            matrix = pd.read_csv(StringIO("\n".join(mat[i + 1:])), sep="\s+", header=None, index_col=0)
            matrix.columns = matrix.index
            matrices.append(matrix)

    return matrices, lengths


def erable_(tree_file, species, msa_files=None, distance_matrices=None, lengths=None, rename=True, naive=False):
    """
    ERaBLE main function
    """
    if not msa_files and not distance_matrices:
        raise ValueError("Provide either distance matrices or MSA files")

    species_mapping = code_species(species)

    A, t = topology_matrix_dijkstra(tree_file, species_mapping)
    # print(A, t, species_mapping)

    # construct C
    C = np.zeros((A.shape[1], A.shape[1]))
    weights = []
    Aks = []
    z = []
    msas = []
    Nks = []
    delta_mats = []

    if not msa_files:
        msa_files = distance_matrices

    for k in range(len(msa_files)):
        if not distance_matrices:
            msa = read_msa(msa_files[k])
            delta_matrix = collapse_on_species(distance_matrix(msa), species)
        else:
            delta_matrix = distance_matrices[k]

        if rename:
            delta_matrix.index = [species_mapping[x] for x in delta_matrix.index]
            delta_matrix.columns = list(delta_matrix.index)
        delta_matrix.sort_index(0, inplace=True)
        delta_matrix.sort_index(1, inplace=True)

        if delta_matrix is not None and len(delta_matrix) > 1:
            delta_mats.append(delta_matrix)
            msas.append(msa_files[k])

    D = np.zeros((len(delta_mats), len(delta_mats)))
    B = np.zeros((A.shape[1], len(delta_mats)))

    for k in range(len(delta_mats)):
        delta = vectorize_delta(delta_mats[k]).reshape(-1, 1)
        Ak = sub_topology_matrix(set(delta_mats[k].index), A)
        if lengths:
            Nk = lengths[k]
            Nks.append(Nk)
        else:
            Nk = sequence_length(msas[k])
            Nks.append(Nk)
        Wk = weight_matrix(Nk, len(delta))
        C += Ak.T @ Wk @ Ak
        weights.append(Wk)
        Aks.append(Ak)
        D[k, k] = delta.T @ Wk @ delta
        B[:, k] = list(-Ak.T @ Wk @ delta)
        Zk = Nk * delta.sum()
        z.append(Zk)

    Z = sum(z)
    z = np.array(z).reshape(-1, 1)

    # return D, B, z, C, Z

    if naive:
        b, alpha = solve_naively(D, B, C, z, Z)

    else:
        b, alpha = solve_cleverly(D, B, C, z, Z)

    # calculate scaling factor
    c = (1 / sum(Nks)) * sum([Nks[i] / alpha[i] for i in range(len(Nks))])

    # rescale
    r = 1 / (c * alpha)
    b = c * b

    return (b, r, c, t)


def erable(tree_file, species_mapping, distance_matrices, lengths, naive=False, subset=False):
    """
    ERaBLE main function
    """
    A, t = topology_matrix_dijkstra(tree_file, species_mapping)
    # print(A, t, species_mapping)

    # construct C
    C = np.zeros((A.shape[1], A.shape[1]))
    weights = []
    Aks = []
    z = []
    Nks = []

    for k in range(len(distance_matrices)):
        distance_matrices[k].sort_index(0, inplace=True)
        distance_matrices[k].sort_index(1, inplace=True)

    D = np.zeros((len(distance_matrices), len(distance_matrices)))
    B = np.zeros((A.shape[1], len(distance_matrices)))

    for k in range(len(distance_matrices)):
        if subset:
            taxa = [species_mapping[x.name] for x in Tree(tree_file).get_leaves()]
            sub = set(taxa) & set(distance_matrices[k].index)
            dist = distance_matrices[k][list(sub)]
            dist = dist.ix[list(sub)]
        else:
            dist = distance_matrices[k]
        delta = vectorize_delta(dist).reshape(-1, 1)
        Ak = sub_topology_matrix(set(dist.index), A)
        Nk = lengths[k]
        Nks.append(Nk)
        Wk = weight_matrix(Nk, len(delta))
        C += Ak.T @ Wk @ Ak
        weights.append(Wk)
        Aks.append(Ak)
        D[k, k] = delta.T @ Wk @ delta
        B[:, k] = list(-Ak.T @ Wk @ delta)
        Zk = Nk * delta.sum()
        z.append(Zk)

    Z = sum(z)
    z = np.array(z).reshape(-1, 1)

    # return D, B, z, C, Z

    if naive:
        b, alpha = solve_naively(D, B, C, z, Z)

    else:
        b, alpha = solve_cleverly(D, B, C, z, Z)

    # calculate scaling factor
    c = (1 / sum(Nks)) * sum([Nks[i] / alpha[i] for i in range(len(Nks))])

    # rescale
    r = 1 / (c * alpha)
    b = c * b

    return (b, r, c, t)