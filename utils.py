import os


def rename_masks(dir_path):
    filenames = os.listdir(dir_path)
    for filename in filenames:
        path = dir_path + '/' + filename
        new_path = path[:-4] + '_mask' + path[-4:]
        os.rename(path, new_path)


if __name__ == '__main__':
    rename_masks('data/2d_masks')