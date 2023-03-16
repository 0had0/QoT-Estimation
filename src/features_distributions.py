import numpy as np
from matplotlib import pyplot as plt
import os, shutil
from tqdm import tqdm
from datasets import load_dataset


def preprocess(data):
    return data.drop(["conn_id", "src_id", "dst_id"], axis=1)


def get_occurrences_per_feature(data, target):
    print(f'[Computing Occurrences] starting')

    occurrences_per_feature = {}

    def extract_features(instance, feature_id):
        occurrences_per_feature[feature_id] = {0: [], 1: []}
        counter_sample = 0

        for _ in instance[feature_id]:
            curr_x = instance[feature_id][counter_sample]
            curr_y = target['class'][counter_sample]
            counter_sample += 1

            rv = np.random.binomial(1, 0.1)
            if rv > 0:
                if curr_y ==  0:
                    occurrences_per_feature[feature_id][0].append(curr_x)
                elif curr_y == 1:
                    occurrences_per_feature[feature_id][1].append(curr_x)

    for index, feature_id in tqdm(enumerate(data.columns), desc="data features", ascii=True, total=len(data.columns)):
        extract_features(data, feature_id)

    for index, feature_id in tqdm(enumerate(target.columns), desc="target features", ascii=True, total=len(target.columns)):
        if feature_id != 'class':
            extract_features(target, feature_id)

    print(f'\n[Computing Occurrences]: all done')

    return occurrences_per_feature


def get_feature_distribution(occurrences_per_feature, feature_id):
    where_are_zeros, where_are_ones = occurrences_per_feature[feature_id][0], occurrences_per_feature[feature_id][1]

    plt.title(feature_id)
    plt.hist(where_are_zeros, bins=50, label='0', alpha=0.5)
    plt.hist(where_are_ones, bins=50, label='1', alpha=0.5)
    plt.legend()

    plt.savefig(f'./features_distributions/{feature_id}.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    if os.path.exists('./features_distributions'):
        shutil.rmtree('./features_distributions')

    os.mkdir('./features_distributions')

    data, target = load_dataset('../datasets/lightpath_dataset_1.nc')
    data = preprocess(data)

    occurrences_per_feature = get_occurrences_per_feature(data, target)

    for feature_id in occurrences_per_feature:
        print(f'[plotting: {feature_id}]\n')
        get_feature_distribution(occurrences_per_feature, feature_id)
