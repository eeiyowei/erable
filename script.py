#!/usr/bin/python3.5
"""
Arthur Zwaenepoel
"""
from erable import erable, parse_distance_matrices, code_species
from ete3 import Tree
import matplotlib.pyplot as plt
import timeit


def benchmark_m():
    print('Reading matrices')
    matrices, lengths = parse_distance_matrices('../distances/OrthoMaM_data_set/inputmatrices')

    tree_file = './data/tree_mam'
    tree = Tree(tree_file)
    species = [n.name for n in tree.get_leaves()]

    print("Coding taxa")
    species_mapping = code_species(species)
    for delta_matrix in matrices:
        delta_matrix.index = [species_mapping[x] for x in delta_matrix.index]
        delta_matrix.columns = list(delta_matrix.index)

    print('Taxa:')
    for key, val in sorted(species_mapping.items()):
        print('{:>20} ---> {}'.format(key, val))

    print('Performing ERaBLE')
    print('\t~~~~~~~~~')

    sizes = [100,200,300,400,500,600,700,800,900,1000,1250,1500,2000]
    times = []
    for size in sizes:
        print('\tm = {:>5}'.format(size))
        t = timeit.timeit(lambda: erable(tree_file=tree_file, species_mapping=species_mapping,
                                         distance_matrices=matrices[:size], lengths=lengths, naive=True),
                          number=1)
        print('\tt = {:>5.1f}'.format(t))
        print('\t---------')
        times.append(t)

    plt.plot(sizes, times)
    plt.savefig('performance_naive.png', dpi=300)


def benchmark_n():
    print('Reading matrices')
    matrices, lengths = parse_distance_matrices('../distances/OrthoMaM_data_set/inputmatrices')

    tree_file = './data/tree_mam'
    tree = Tree(tree_file)
    species = [n.name for n in tree.get_leaves()]

    print("Coding taxa")
    species_mapping = code_species(species)
    for delta_matrix in matrices:
        delta_matrix.index = [species_mapping[x] for x in delta_matrix.index]
        delta_matrix.columns = list(delta_matrix.index)

    print('Taxa:')
    for key, val in sorted(species_mapping.items()):
        print('{:>20} ---> {}'.format(key, val))

    print('Performing ERaBLE')
    print('\t~~~~~~~~~')

    sizes = [5,6,7,8,9,11,12,14,15,16,17,18]
    ns = []
    times = []
    for size in sizes:
        n = tree.get_leaves_by_name('Homo')[0]
        for i in range(size):
            n = n.up
        n.write(outfile='tmp.tree')

        print('\tn = {:>5}'.format(size))
        print(n)
        t = timeit.timeit(lambda: erable(tree_file='tmp.tree', species_mapping=species_mapping,
                                         distance_matrices=matrices[:5000], lengths=lengths,
                                         subset=True),
                          number=1)
        ns.append(len(n))
        print('\tt = {:>5.1f}'.format(t))
        print('\t---------')
        times.append(t)

    plt.plot(ns, times)
    plt.savefig('performance_n.png', dpi=300)





"""
print('Scale factor: {0}\n{1}'.format(c, '-'*70))

print('Branch lengths:')
print(b)
print('-'*70)

print('Gene rates:')
print(r)
"""

if __name__ == '__main__':
    benchmark_n()