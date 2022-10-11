import numpy as np
# Helper function to saver classifier list or classifier name
def save_list(my_list, filename):
    np.save(filename, my_list)
    print("saved successfuly")



def load_list(filename):
    file=np.load(filename)
    return file.tolist()