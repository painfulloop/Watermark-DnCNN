import os

def create_folder(path_folder):
    try:
        os.makedirs(path_folder)
    except FileExistsError:
        print('directory {} already exist'.format(path_folder))
        pass
    except OSError:
        print('creation of the directory {} failed'.format(path_folder))
        pass
    else:
        print("Succesfully created the directory {} ".format(path_folder))
    return path_folder


def get_last_model(path:str):
    _models = [file[:-len('.ckpt.meta')] for file in sorted(os.listdir(path)) if file.endswith('.ckpt.meta')]
    return _models[-1]