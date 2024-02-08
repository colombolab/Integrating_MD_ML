import numpy as np
from pandas import read_csv, DataFrame, concat
from glob import glob
from os.path import splitext, basename
from os import sep
from rdkit.Chem.AllChem import MolFromMolFile, GetMorganFingerprintAsBitVect, MolToSmiles


def extract_rmsd(path):
    rmsd_a, rmsd_b = DataFrame(), DataFrame()
    for path_configuration in glob(path):
        for path_structure in glob(path_configuration + '\\*'):
            structure = splitext(basename(path_structure))[0]
            if basename(path_configuration) == 'Backbone':
                rmsd_b[structure] = read_csv(path_structure, delim_whitespace=True).iloc[:, 1]
            else:
                rmsd_a[structure] = read_csv(path_structure, delim_whitespace=True).iloc[:, 1]
    return concat([rmsd_a, rmsd_b], keys=('all', 'backbone'))


def extract_distance(path, site):
    distances, couples = list(), list()
    for dist in glob(path):
        for path_binding_site in glob(dist + '\\*'):
            if basename(path_binding_site) == site:
                distances.append(extract_rmsd(path_binding_site + '\\*'))
        couples.append(splitext(basename(dist))[0].split('_')[1])
    return concat(distances, keys=couples)


def extract_df_simulation(path):
    dfs, simulations = list(), list()
    for path_binding_site in glob(path):
        simulations.append(basename(path_binding_site))
        df = DataFrame()
        for path_structure in glob(path_binding_site + '\\*'):
            structure = splitext(basename(path_structure))[0]
            if path.split(sep)[1] == 'Distance Fluctuations - Separate':
                df[structure] = read_csv(path_structure, delim_whitespace=True, header=None).iloc[:, 2]
            else:
                df[structure] = read_csv(path_structure, delim_whitespace=True, header=None).iloc[:, 1]
        dfs.append(df)
    return concat(dfs, keys=simulations)


def extract_ecfp_and_smiles(path, r=2, bits=2 ** 10, use_features=True, use_chirality=True):
    fingerprint, smile = DataFrame(), DataFrame()
    for path_structure in glob(path):
        structure = splitext(basename(path_structure))[0]
        molecule = MolFromMolFile(path_structure)
        fps = GetMorganFingerprintAsBitVect(
            molecule, radius=r, nBits=bits, useFeatures=use_features, useChirality=use_chirality)
        fingerprint[structure] = np.array(fps)
        smile[structure] = [MolToSmiles(molecule)]
    return fingerprint, smile


def save_file(feature, filename, save, path):
    site = list(feature.keys())[::-1]
    feature_ortho = feature[site[0]]
    if save:
        feature_ortho.to_parquet('Data\\{}'.format(path) + filename + '_orthosteric.parquet')
    if len(site) == 2:
        feature_allo = feature[site[1]]
        if save:
            feature_allo.to_parquet('Data\\{}'.format(path) + filename + '_allosteric.parquet')


def feature_extraction(chosen_feature, save):
    feature = {}
    path = input('Default path (yes/no): ')
    path = ('' if path == 'yes' else 'Test\\Data - Orthosteric\\')
    if chosen_feature == 'rmsd':
        if len(path) == 0:
            feature['rmsd_allo'] = extract_rmsd('Data\\RMSD\\Allosteric\\*'.format(path))
        feature['rmsd_ortho'] = extract_rmsd('Data\\{}RMSD\\Orthosteric\\*'.format(path))
        save_file(feature, 'rmsd', save, path)
    elif chosen_feature == 'distance':
        if len(path) == 0:
            feature['dist_allo'] = extract_distance('Data\\Distances\\*', 'Allosteric')
        feature['dist_ortho'] = extract_distance('Data\\{}Distances\\*'.format(path), 'Orthosteric')
        save_file(feature, 'dist', save, path)
    elif chosen_feature == 'df':
        if len(path) == 0:
            feature['df_separate_allo'] = extract_df_simulation('Data\\Distance Fluctuations - Separate\\Allosteric\\*')
        feature['df_separate_ortho'] = extract_df_simulation('Data\\{}Distance Fluctuations - Separate\\Orthosteric\\*'.
                                                             format(path))
        if len(path) == 0:
            feature['df_average_allo'] = extract_df_simulation('Data\\Distance Fluctuations - Average\\Allosteric\\*')
        feature['df_average_ortho'] = extract_df_simulation('Data\\{}Distance Fluctuations - Average\\Orthosteric\\*'.
                                                            format(path))
        save_file({key: feature[key] for key in ['df_separate_allo', 'df_separate_ortho']}, 'df_separate', save, path)
        save_file({key: feature[key] for key in ['df_average_allo', 'df_average_ortho']}, 'df_average', save, path)
    elif chosen_feature == 'fingerprints':
        if len(path) == 0:
            feature['fingerprints_allo'] = extract_ecfp_and_smiles('Data\\Ligands\\Allosteric\\*'.format(path))[0]
        feature['fingerprints_ortho'] = extract_ecfp_and_smiles('Data\\{}Ligands\\Orthosteric\\*'.format(path))[0]
        if len(path) == 0:
            feature['smiles_allo'] = extract_ecfp_and_smiles('Data\\Ligands\\Allosteric\\*'.format(path))[1]
        feature['smiles_ortho'] = extract_ecfp_and_smiles('Data\\{}Ligands\\Orthosteric\\*'.format(path))[1]
        save_file({key: feature[key] for key in ['fingerprints_allo', 'fingerprints_ortho']}, 'fingerprints',
                  save, path)
        save_file({key: feature[key] for key in ['smiles_allo', 'smiles_ortho']}, 'smiles', save, path)
    else:
        pass
    return feature


if __name__ == "__main__":
    features = str(input('Enter features to extract (rmsd/distance/df/fingerprints): '))
    file = input('Save features file (True/False): ') == 'True'
    features = feature_extraction(features, file)
