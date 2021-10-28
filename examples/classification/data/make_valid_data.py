"""Make the separate validation data, so that we don't tune on dev set.

python -m classification.data.make_valid_data
"""
import os

import fire
import numpy as np
import tqdm


def write_lines(path, lines, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        f.writelines(lines)
        print(len(lines))


def main():
    valid_percentage = 0.1
    original_dir = "/nlp/scr/lxuechen/data/lm-bff/data/original"
    new_dir = "/nlp/scr/lxuechen/data/lm-bff/data/glue-with-validation"

    task_folders = ("GLUE-SST-2", "QNLI", "QQP")
    for task_folder in task_folders:
        # Create train and valid splits.
        full_train_path = os.path.join(original_dir, task_folder, 'train.tsv')
        with open(full_train_path, 'r') as f:
            full_train = f.readlines()

        header = full_train[0]
        full_train = full_train[1:]  # Remove header.

        indices = np.random.permutation(len(full_train))
        new_valid_size = int(len(indices) * valid_percentage)
        new_train_size = len(indices) - new_valid_size
        new_train_indices = indices[:new_train_size]
        new_valid_indices = indices[new_train_size:]
        assert len(new_train_indices) == new_train_size
        assert len(new_valid_indices) == new_valid_size

        new_train = [header] + [full_train[i] for i in new_train_indices]
        new_valid = [header] + [full_train[i] for i in new_valid_indices]

        new_train_path = os.path.join(new_dir, task_folder, 'train.tsv')
        new_valid_path = os.path.join(new_dir, task_folder, 'dev.tsv')

        write_lines(new_train_path, new_train)
        write_lines(new_valid_path, new_valid)
        del new_train, new_valid, new_train_path, new_valid_path
        del new_train_size, new_train_indices
        del new_valid_size, new_valid_indices

        # Make test!
        test_path = os.path.join(original_dir, task_folder, 'dev.tsv')
        new_test_path = os.path.join(new_dir, task_folder, 'test.tsv')
        os.system(f'cp {test_path} {new_test_path}')
        del test_path, new_test_path

    # Make valid set for MNLI; different, since matched/mismatched!
    task_folder = "MNLI"
    matched_genres = ['slate', 'government', 'telephone', 'travel', 'fiction']
    mismatched_genres = ['letters', 'verbatim', 'facetoface', 'oup', 'nineeleven']
    full_train_path = os.path.join(original_dir, task_folder, 'train.tsv')
    with open(full_train_path, 'r') as f:
        full_train = f.readlines()
        full_train_csv = [line.split('\t') for line in full_train]

        # Check the lengths are correct.
        l = len(full_train_csv[0])
        for line in full_train_csv:
            assert l == len(line)

    # Remove header.
    header = full_train[0]
    header_csv = full_train_csv[0]

    full_train = full_train[1:]
    full_train_csv = full_train_csv[1:]

    # Get index of genre.
    genre_index = header_csv.index('genre')

    # Shuffle both!
    indices = np.random.permutation(len(full_train))
    full_train = [full_train[i] for i in indices]
    full_train_csv = [full_train_csv[i] for i in indices]

    # Split validation.
    new_valid_size = int(len(indices) * valid_percentage)
    new_matched_valid_size = new_mismatched_valid_size = new_valid_size // 2

    # Fetch the indices.
    new_train_indices = []
    new_matched_valid_indices = []
    new_mismatched_valid_indices = []
    matched_count = mismatched_count = 0
    for i, row in enumerate(full_train_csv):
        genre = row[genre_index]
        if genre in matched_genres and matched_count < new_matched_valid_size:
            new_matched_valid_indices.append(i)
            matched_count += 1
        elif genre in mismatched_genres and mismatched_count < new_mismatched_valid_size:
            new_mismatched_valid_indices.append(i)
            mismatched_count += 1
        else:
            new_train_indices.append(i)

    new_matched_valid_indices = set(new_matched_valid_indices)
    new_mismatched_valid_indices = set(new_mismatched_valid_indices)

    new_train = [header]
    new_matched_valid = [header]
    new_mismatched_valid = [header]
    for i, line in tqdm.tqdm(enumerate(full_train)):
        if i in new_matched_valid_indices:
            new_matched_valid.append(line)
        elif i in new_mismatched_valid_indices:
            new_mismatched_valid.append(line)
        else:
            new_train.append(line)

    new_train_path = os.path.join(new_dir, task_folder, 'train.tsv')
    new_matched_valid_path = os.path.join(new_dir, task_folder, 'dev_matched.tsv')
    new_mismatched_valid_path = os.path.join(new_dir, task_folder, 'dev_mismatched.tsv')

    write_lines(new_train_path, new_train)
    write_lines(new_matched_valid_path, new_matched_valid)
    write_lines(new_mismatched_valid_path, new_mismatched_valid)

    matched_test_path = os.path.join(original_dir, task_folder, 'dev_matched.tsv')
    new_matched_test_path = os.path.join(new_dir, task_folder, 'test_matched.tsv')
    os.system(f'cp {matched_test_path} {new_matched_test_path}')

    mismatched_test_path = os.path.join(original_dir, task_folder, 'dev_mismatched.tsv')
    new_mismatched_test_path = os.path.join(new_dir, task_folder, 'test_mismatched.tsv')
    os.system(f'cp {mismatched_test_path} {new_mismatched_test_path}')


if __name__ == "__main__":
    fire.Fire(main)
