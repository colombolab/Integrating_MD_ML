import numpy as np
from warnings import warn
from pandas import DataFrame, concat, MultiIndex, read_parquet
from os.path import dirname


np.random.seed(123)
keys = ['removing_predictor', 'predictor_removed', 'removed', 'index_distance_removed', 'not_removed']
setting = {key: None for key in keys}
keys = ['structure', 'save', 'first', 'second']
testing = {key: None for key in keys}
number = {'frame': 2000, 'couple': 15}
features_expand = []


def extract_frame(predictor, n_frame, init_frame, period=True):
    global number
    if period:
        period = (int(len(predictor) / number['couple']) if predictor.index.nlevels == 2 else len(predictor)) // n_frame
        chosen_index = [init_frame + frame * period for frame in range(n_frame)]
    else:
        chosen_index = np.sort(np.random.choice(number['frame'], replace=False, size=n_frame))
    if predictor.index.nlevels == 2:
        new_index = MultiIndex.from_tuples([(level0, level1 % number['frame']) for level0, level1 in predictor.index])
        predictor.index = new_index
        return predictor.loc[(slice(None), chosen_index), :]
    return predictor.iloc[chosen_index]


def time_series(predictor, n_frame, init_frame, random, period):
    predictor_period = ([extract_frame(predictor[i], n_frame, init_frame) for i in range(len(predictor))]
                        if period else list())
    predictor_random = ([extract_frame(predictor[i], n_frame, init_frame, period=False) for i in range(len(predictor))]
                        if random else list())
    return predictor_period + predictor_random


def ask_setting(setting_predictor):
    distance = ['D-E', 'E-L', 'I-D', 'I-E', 'I-K', 'I-L', 'K-D', 'K-E', 'K-L', 'K-V', 'L-D', 'V-D', 'V-E', 'V-I', 'V-L']
    if setting_predictor['removing_predictor'] is None:
        remove = input('Do you want to remove any predictor? (Y/n): ').lower()
        setting_predictor['removing_predictor'] = remove
    if setting['removing_predictor'] == 'y':
        if setting_predictor['predictor_removed'] is None:
            remove = input(' + Enter the predictor to remove (rmsd/distance/df/fingerprints): ').lower()
            setting_predictor['predictor_removed'] = remove
        if setting_predictor['predictor_removed'] in ['rmsd', 'df', 'fingerprints']:
            if setting['removed'] is None:
                print('\n   * Setting: features without {} predictor \n'.format(setting['predictor'].upper()))
                setting['removed'] = True
        if setting_predictor['predictor_removed'] == 'distance':
            if setting['index_distance_removed'] is None:
                index = input(' + Enter index of couple (1/15): ')
                setting['index_distance_removed'] = index
                print('\n   * Setting: features without Distance {} predictor \n'.format(
                    distance[int(setting['index_distance_removed'])]))
    else:
        if setting['not_removed'] is None:
            print('\n   * Setting: features with all predictors \n')
            setting['not_removed'] = True


def remove_predictor(rmsd, dist, df_s, df_a, fing, resp):
    global setting
    ask_setting(setting)
    if setting['removing_predictor'] == 'y':
        while True:
            ask_setting(setting)
            if setting['predictor_removed'] == 'rmsd':
                ask_setting(setting)
                data = [concat([dist[i], df_s[i], df_a[i], fing, resp.T], ignore_index=True).T
                        for i in range(len(rmsd))]
                break
            elif setting['predictor_removed'] == 'distance':
                ask_setting(setting)
                couple = dist[0].index.get_level_values(0).unique()[int(setting['index_distance_removed']) - 1]
                dist = [dist[i].drop(couple, level=0) for i in range(len(dist))]
                data = [concat([rmsd[i], dist[i], df_s[i], df_a[i], fing, resp.T], ignore_index=True).T
                        for i in range(len(rmsd))]
                break
            elif setting['predictor_removed'] == 'df':
                ask_setting(setting)
                data = [concat([rmsd[i], dist[i], fing, resp.T], ignore_index=True).T
                        for i in range(len(rmsd))]
                break
            elif setting['predictor_removed'] == 'fingerprints':
                ask_setting(setting)
                data = [concat([rmsd[i], dist[i], df_s[i], df_a[i], resp.T], ignore_index=True).T
                        for i in range(len(rmsd))]
                break
            else:
                warn('You may have typed incorrectly, please enter the predictor to remove again.')
                setting['predictor_removed'] = None
    else:
        ask_setting(setting)
        data = [concat([rmsd[i], dist[i], df_s[i], df_a[i], fing, resp.T], ignore_index=True).T
                for i in range(len(rmsd))]
    return data


def union_predictors(n_frame, init_frame, random, period, features):
    global number
    data_binding = []
    for file_prefix in (['orthosteric'] if features else ['allosteric', 'orthosteric']):
        if file_prefix == 'allosteric':
            resp = [0]
        else:
            resp = [1]

        rmsd = read_parquet('Data\\{}rmsd_{}.parquet'.format('Test\\Data - Orthosteric\\' if features else '',
                                                             file_prefix))
        rmsd = [rmsd.xs(atom, level=0) for atom in ['all', 'backbone']]
        rmsd = [rmsd[i].loc[slice(j * number['frame'], number['frame'] - 1 + j * number['frame']), :]
                for i in range(2) for j in range(4)]
        rmsd = time_series(rmsd, n_frame, init_frame, random, period)

        dist = read_parquet('Data\\{}dist_{}.parquet'.format('Test\\Data - Orthosteric\\' if features else '',
                                                             file_prefix))
        dist = [dist.xs(atom, level=1) for atom in ['all', 'backbone']]
        dist = [dist[i].loc[(slice(None), slice(j * number['frame'], number['frame'] - 1 + j * number['frame'])), :]
                for i in range(2) for j in range(4)]
        dist = time_series(dist, n_frame, init_frame, random, period)

        df_s = read_parquet('Data\\{}df_separate_{}.parquet'.format('Test\\Data - Orthosteric\\' if features else '',
                                                                    file_prefix))
        df_s = [df_s.xs(sim) for sim in df_s.index.get_level_values(0).unique()] * (4 if random and period else 2)

        df_a = read_parquet('Data\\{}df_average_{}.parquet'.format('Test\\Data - Orthosteric\\' if features else '',
                                                                   file_prefix))
        df_a = [df_a.xs(sim) for sim in df_a.index.get_level_values(0).unique()] * (4 if random and period else 2)

        fing = read_parquet('Data\\{}fingerprints_{}.parquet'.format('Test\\Data - Orthosteric\\' if features else '',
                                                                     file_prefix))

        resp = DataFrame({'y': resp*len(rmsd[0].columns)}, index=rmsd[0].columns)

        data = remove_predictor(rmsd, dist, df_s, df_a, fing, resp)
        data_binding.append(concat(data, ignore_index=False))
    if features:
        return data_binding[0]
    else:
        return data_binding[0], data_binding[1]


def test_allosteric(data_allosteric, features):
    global testing, features_expand
    if testing['structure'] is None:
        testing['structure'] = int(input(' + Enter which structure to test ({}): '.format('(1/5)' if features
                                                                                          else '(1/19)')))
    i = testing['structure']
    residue = ['1w0x', '2euf', '3py0', '5l2s', '5l2t'] if features \
        else ['3ezr', '3pxf', '4d1z', '4fku', '5a14', '5fp5', '5fp6', '5oo0', '5osj', '6q3f', '6q49', '6q4k_first',
              '6q4k_second', '7rwe', '7rwf', '7rxo', '7s4t', '7s84', '7s85']
    drop_index = residue[i - 1]
    print('\n   * Setting: residue {} to test \n'.format(residue[i - 1]))
    features_expand.append(data_allosteric.loc[drop_index, :])
    if testing['save']:
        testing['save'] = input('Do you want to save the data removed? (Y/n): ') == 'y'
    save = testing['save']
    if save:
        concat(features_expand, ignore_index=False).to_parquet('Data\\Test\\features{}.parquet'.format(i))
        path = dirname('Data\\Test\\features{}.parquet'.format(i))
        print('\n   * features{}.parquet is saved in folder {} \n'.format(i, path))
    if i == 12:
        if testing['second'] is None:
            testing['second'] = input('Do you want to remove also 6q4k_second? (Y/n): ').lower() == 'y'
        double_drop = testing['second']
        if double_drop:
            print('\n   ** : residue 6q4k_second is removed \n')
            drop_index = [drop_index, residue[i]]
    if i == 13:
        if testing['first'] is None:
            testing['first'] = input('Do you want to remove also 6q4k_first? (Y/n): ').lower() == 'y'
        double_drop = testing['first']
        if double_drop:
            print('\n   ** : residue 6q4k_first is removed \n')
            drop_index = [drop_index, residue[i - 2]]
    data_allosteric = data_allosteric.drop(drop_index)
    return data_allosteric


def extract_features_and_output(n_frame, start, random, period, drop, features):
    if features:
        data_allosteric = union_predictors(n_frame, start, random, period, features)
    else:
        data_allosteric, data_orthosteric = union_predictors(n_frame, start, random, period, features)
    data_allosteric = (test_allosteric(data_allosteric, features) if drop == 'y' else data_allosteric)
    data_allosteric = data_allosteric.reset_index(drop=True)
    if not features:
        data_orthosteric = data_orthosteric.reset_index(drop=True)
    data = (concat([data_allosteric]) if features else concat([data_allosteric, data_orthosteric]))
    data = data.sample(n=len(data), random_state=42, ignore_index=True)
    xdata = data.iloc[:, :-1]
    ydata = data.iloc[:, -1]
    return xdata, ydata


def create_dataset(n_frame, expand=False):
    global number, testing
    period = input('Do you want to increase the dataset by extracting periodic frames? (Y/n): ') == 'y'
    if period:
        expand = input('Do you want to increase the dataset by extracting all periodic frames? (Y/n): ') == 'y'
    if expand:
        random = False
        print('\n   ** Setting: all periodic frames are extracted. \n')
    else:
        random = input('Do you want to increase the dataset by extracting random frames? (Y/n): ') == 'y'
        if not random and not period:
            period = True
            warn('Random and period parameters cannot both be False. Period parameter is set to True.')
        print('\n   * Setting: Periodic parameter - {}, Random parameter - {} \n'.format(period, random))
    features = input('Do you want to extract features for the orthosteric structures? (Y/n): ') == 'y'
    if not features:
        drop = str(input('Do you want to remove from dataset any allosteric ligand to test? (Y/n): ')).lower()
        if drop == 'n':
            print('\n   * Setting: no one allosteric ligand to test \n')
    else:
        drop = 'y'
    xdata, ydata = DataFrame(), DataFrame()
    if expand:
        for start in range(number['frame'] // n_frame - 1):
            if start == number['frame'] // n_frame - 2:
                testing['save'] = True
            xdat, ydat = extract_features_and_output(n_frame, start, random, period, drop, features)
            xdata = concat([xdata, xdat])
            ydata = concat([ydata, ydat])
    else:
        start = int(np.random.choice(number['frame'] // n_frame - 1))
        testing['save'] = True
        xdat, ydat = extract_features_and_output(n_frame, start, random, period, drop, features)
        xdata = concat([xdata, xdat])
        ydata = concat([ydata, ydat])
    print('   ** Setting: the dataset has {} data with {} features \n'.format(xdata.shape[0], xdata.shape[1]))
    data = concat([xdata, ydata], axis=1, ignore_index=True)
    data = data.sample(n=len(data), random_state=42, ignore_index=True)
    xdata = data.iloc[:, :-1]
    ydata = data.iloc[:, -1]
    return xdata, ydata


def save_dataset(xdata, ydata):
    save_data = input('Do you want to save the dataset (Y/n): ').lower()
    if save_data == 'y':
        name = input(' + Enter the name of file .parquet to save')
        concat([xdata, ydata], axis=1).to_parquet('Data\\Test\\{}.parquet'.format(name))
        path = dirname('Data\\Test\\{}.parquet'.format(name))
        print('\n   * {}.parquet is saved in folder {} \n'.format(name, path))


if __name__ == "__main__":
    x, y = create_dataset(80)
    save_dataset(x, y)
