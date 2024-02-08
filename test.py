import numpy as np
from pickle import load
from pandas import read_parquet, concat, DataFrame
from sklearn.impute import KNNImputer
from rdkit.Chem.AllChem import MolFromMolFile, GetMorganFingerprintAsBitVect
from glob import glob
from os.path import splitext, basename
from warnings import warn
from scipy.stats import pearsonr, kendalltau, spearmanr


n_fingerprints = 1024


def create_test(filename, n_features):
    global n_fingerprints
    molecule = MolFromMolFile(filename)
    fps = GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=n_fingerprints, useFeatures=True, useChirality=True)
    fps = list(fps)
    return DataFrame([None] * (n_features - len(fps)) + fps).T.dropna(axis='columns')


def imputation(filename, n_structure, binding_site, double, database):
    data = read_parquet('Data\\Test\\Datasets\\{}\\dataset{}.parquet'.
                        format(database, 1213 if double else n_structure)
                        if binding_site == 'Allosteric' else 'Data\\Test\\Datasets\\{}\\dataset.parquet'.
                        format(database))
    x = data.iloc[:, :-1]
    test_data = create_test(filename, x.shape[1])
    test_data = concat([DataFrame([None] * (x.shape[1] - test_data.shape[1])).T, test_data], axis=1)
    x = concat([x, test_data])
    imputer = KNNImputer(n_neighbors=5)
    x_imputed = imputer.fit_transform(x)
    return DataFrame(x_imputed[-1, :]).T


def similarity(test_data, n_structure, binding_site, database):
    global n_fingerprints
    features = read_parquet('Data\\Test\\Features\\{}\\{}\\features{}.parquet'.
                            format(binding_site, database, n_structure))
    n_features = test_data.shape[1] - n_fingerprints
    correlation, name = [pearsonr, kendalltau, spearmanr], ['Pearson', 'Kendall', 'Spearman']
    for j, corr in enumerate(correlation):
        simil = [corr(features.iloc[i, :n_features].values, test_data.iloc[0, :n_features].values)[0]
                 for i in range(features.shape[0])]
        print('Level of similarity ({}): {}'.format(name[j], round(DataFrame(simil).mean().values[0], 2)))


def prediction(structure, model, binding_site, n_structure, double, database):
    test_data = imputation('Data\\Test\\{}\\{}.mol'.format(binding_site, structure), n_structure,
                           binding_site, double, database)
    if binding_site in ['Allosteric', 'Orthosteric']:
        similarity(test_data, n_structure, binding_site, database)
    predict = model.predict(test_data)
    probability = round(max(model.predict_proba(test_data)[0]) * 100, 2)
    print('\n{} is allosteric with a {}% probability'.format(structure, probability) if predict == 0
          else '\n{} is orthosteric with a {}% probability'.format(structure, probability))
    if (binding_site in ['Allosteric', 'Allosteric Kinase', 'Allosteric New'] and predict == 1) or \
       (binding_site in ['Orthosteric', 'Orthosteric Kinase', 'Orthosteric New'] and predict == 0):
        warn('Misclassified!')


def feature_importance(trained_model, model):
    global n_fingerprints
    if trained_model.split('_')[0] == 'RF':
        importance = round(model.feature_importances_[:-n_fingerprints].sum() * 100, 2)
        print('Feature importance: dynamical - {}%, fingerprints - {}% \n'.
              format(importance, round(100 - importance, 2)))
    if trained_model.split('_')[0] == 'SVM':
        support_vectors = model.support_vectors_
        dual_coefficients = np.abs(model.dual_coef_)
        importance = np.dot(dual_coefficients, support_vectors)[0]
        importance = round((importance / importance.sum())[:-n_fingerprints].sum() * 100, 2)
        print('Feature importance: dynamical - {}%, fingerprints - {}% \n'.
              format(importance, round(100 - importance, 2)))
    if trained_model.split('_')[0] == 'MLP':
        weights = model.coefs_[0]
        dynamical = weights[:-n_fingerprints]
        importance = round(np.absolute(dynamical).sum() / np.absolute(weights).sum() * 100, 2)
        print('Feature importance: dynamical - {}%, fingerprints - {}% \n'.
              format(importance, round(100 - importance, 2)))


def test():
    binding_site = str(input('Which binding site do you want to test? (Allosteric/Orthosteric (-/Kinase/New)): '))
    structures = [splitext(basename(name))[0] for name in glob('Data\\Test\\{}\\*'.format(binding_site))]
    n_structure = input(' + Enter which structure to test ({}): '.format(
        '1/19' if binding_site == 'Allosteric' else '1/5' if binding_site == 'Orthosteric' else '1/6 or all'))
    double = False
    if binding_site == 'Allosteric' and (int(n_structure) == 12 or int(n_structure) == 13):
        double = input('\nDo you want to test {} without {}? (Y/n)'.format(
            '6q4k_first' if int(n_structure) == 11 else '6q4k_second',
            '6q4k_second' if int(n_structure) == 11 else '6q4k_first')).lower() == 'y'
    print('\n   * Setting: all residue to test \n' if n_structure == 'all'
          else '\n   * Setting: residue {} to test \n'.format(structures[int(n_structure) - 1]))
    database = input('Which dataset do you want to test? (base/complete/extended)')
    model = 'MLP' if database == 'extended' else input('Which trained model do you want to use for test? (RF/SVM): ')
    suffix = {'RF': '_100-5', 'SVM': '_0.2'}
    trained_model = str(model).upper() + suffix.get(model, '')
    path = ('Data\\Test\\Models\\{}\\{}_{}.pkl'.format(database, trained_model, 1213)
            if binding_site == 'Allosteric' and double
            else 'Data\\Test\\Models\\{}\\{}_{}.pkl'.format(database, trained_model, n_structure)
            if binding_site == 'Allosteric' else 'Data\\Test\\Models\\{}\\{}.pkl'.format(database, trained_model))
    print('\n   * Setting: model {} used for test \n'.format(basename(path)))
    with open(path, 'rb') as file:
        model = load(file)
    feature_importance(trained_model, model)
    ([prediction(structure, model, binding_site, n_structure, double, database) for structure in structures]
     if n_structure == 'all' and binding_site not in ['Allosteric', 'Orthosteric']
     else prediction(structures[int(n_structure) - 1], model, binding_site, n_structure, double, database))


if __name__ == "__main__":
    test()
