import copy
import shutil
import pickle

from utils_abide import *

num_subjects = 871

atlas = 'ho'

connectivity = 'correlation'

files = ['rois_ho', 'conn_mat']

filemapping = {'rois_ho': 'rois_ho.1D',
               'conn_mat': '_' + atlas + '_' + connectivity + '.mat'}

n_folds = 10
seed_data = 123
prng = np.random.RandomState(seed_data)


def save_mat():
    subject_IDs = get_ids(num_subjects).tolist()

    # Create a folder for each subject

    for s, fname in zip(subject_IDs, fetch_filenames(subject_IDs, files[0])):
        time_series = get_timeseries(fname)

        # Compute and save functional connectivity matrices
        connectivity_mat = subject_connectivity(time_series, s, connectivity)
        sio.savemat(s + filemapping['conn_mat'], {'connectivity': connectivity_mat})

        subject_folder = os.path.join('./', s)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        if not os.path.exists(os.path.join(subject_folder, base + filemapping['rois_ho'])):
            shutil.move(base + filemapping['rois_ho'], subject_folder)
        if not os.path.exists(os.path.join(subject_folder, s + filemapping['conn_mat'])):
            shutil.move(s + filemapping['conn_mat'], subject_folder)


def split_data(site, train_perc):
    """ Split data into training and test indices """
    train_indices = []
    test_indices = []

    for s in np.unique(site):
        if s == -1:
            continue

        # Make sure each site is represented in both training and test sets
        id_in_site = np.argwhere(site == s).flatten()

        num_nodes = len(id_in_site)
        train_num = int(train_perc * num_nodes)

        prng.shuffle(id_in_site)
        train_indices.extend(id_in_site[:train_num])
        test_indices.extend(id_in_site[train_num:])

    return np.array(train_indices), np.array(test_indices)


def split_by_site(idx, site):
    idx = np.array(idx)
    unq = np.unique(site[idx]).tolist()
    idx_single_site = [idx[np.where(site[idx] == s)] for s in unq]
    assert len(idx_single_site) == 20
    return np.array(idx_single_site, dtype=object)


def load_data():
    subject_IDs = get_ids(num_subjects)

    # Get all subject networks
    X = load_all_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Get ROI coordinates
    coords = get_atlas_coords(atlas_name=atlas)

    # Get subject labels
    label_dict = get_subject_score(subject_IDs, score='DX_GROUP')
    y = np.array([int(label_dict[x]) - 1 for x in sorted(label_dict)])

    # Get site ID
    site = get_subject_score(subject_IDs, score='SITE_ID')
    unq = np.unique(list(site.values())).tolist()
    site = np.array([unq.index(site[x]) for x in sorted(site)])

    # Choose site IDs to include in the analysis
    site_mask = range(20)
    X = X[np.in1d(site, site_mask)]
    y = y[np.in1d(site, site_mask)]
    site = site[np.in1d(site, site_mask)]

    # split data by k-fold CV(training set & test set)
    cv_splits = [[[], []] for i in range(10)]  # [ [ [Train], [Test] ], ..., ..., []]
    for s in range(20):
        # 每个站点进行K折交叉验证，然后再拼接在一起
        site_ids = list(np.where(site == s)[0])
        def get_fake_cv_splits(lst, ratio=0.8):
            import random
            cut = int(len(lst) * ratio)
            tmp_lst = []
            for k in range(10):
                random.shuffle(lst)
                fold_train_ids, fold_test_ids = lst[:cut], lst[cut:]
                tmp_lst.append([fold_train_ids, fold_test_ids])
            return tmp_lst
        site_cv_splits = get_fake_cv_splits(site_ids)
        for fold in range(10):
            train_ids = site_cv_splits[fold][0]
            test_ids = site_cv_splits[fold][1]
            cv_splits[fold][0].extend(train_ids)
            cv_splits[fold][1].extend(test_ids)
    # Training, validation, test sets(divide training set into training and validation sets)
    idxs_train = []
    idxs_val = []
    idxs_test = []
    for i in range(len(cv_splits)):
        idx_test = cv_splits[i][1]
        site_mask = copy.deepcopy(site)
        site_mask[idx_test] = -1
        idx_train, idx_val = split_data(site_mask, 1 - len(idx_test) / num_subjects)
        idxs_train.append(idx_train)
        idxs_val.append(idx_val)
        idxs_test.append(idx_test)
        print("\rFold: {}, training size: {}, val size: {}, test size: {}".format(i, len(idx_train), len(idx_val), len(idx_test)))

    bas = (X, y, idxs_train, idxs_val, idxs_test)
    with open('save/bas_cv', 'wb') as f:
        pickle.dump(bas, f)

    y_site = site
    idxs_train_site = []
    idxs_val_site = []
    idxs_test_site = []
    for i in range(n_folds):
        idxs_train_site.append(split_by_site(idxs_train[i], site))
        idxs_val_site.append(split_by_site(idxs_val[i], site))
        idxs_test_site.append(split_by_site(idxs_test[i], site))

    site_info = (y_site, idxs_train_site, idxs_val_site, idxs_test_site)
    with open('save/site_info_cv', 'wb') as f:
        pickle.dump(site_info, f)


if __name__ == '__main__':
    load_data()
