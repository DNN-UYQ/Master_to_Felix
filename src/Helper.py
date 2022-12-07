import numpy as np
from sklearn.metrics import confusion_matrix
# Helper function to saver classifier list or classifier name
def save_list(my_list, filename):
    np.save(filename, my_list)
    print("saved successfuly")



def load_list(filename):
    file= np.load(filename, allow_pickle=True)
    return file.tolist()

def get_confusion_matrix_values(y_true, y_pred):
    cm: object = confusion_matrix(y_true, y_pred)
    TP = cm[0, 0]
    FP = cm[0, 1]
    TN = cm[1, 0]
    FN = cm[1, 1]

    return(TP, FP, TN,FN)



"""a,b,c,d=get_confusion_matrix_values([0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0])
print(d)"""