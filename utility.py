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