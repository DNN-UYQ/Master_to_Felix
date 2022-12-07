import pandas as pd
import numpy as np
#Helper function to load our unified data representation (either for train or test)
def load_unified_data(file_name):
    df_test= pd.read_excel(file_name)
    X_big= df_test.iloc[:,1:-1]
    X_big_test=X_big.to_numpy()
    y_big=df_test.iloc[:,-1]
    y_big_test=y_big.to_numpy()
    return X_big_test, y_big_test



