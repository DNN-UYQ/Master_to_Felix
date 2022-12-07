import sklearn.tree
from pymfe.mfe import MFE

def meta_features_model(X,y):
    model = sklearn.tree.DecisionTreeClassifier().fit(X, y)
    extractor = MFE()
    ft = extractor.extract_from_model(model)
    return ft





    #print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
    """# Extract specific metafeatures from model
    extractor = MFE(features=["tree_shape", "nodes_repeated"], summary="histogram")

    ft = extractor.extract_from_model(
        model,
        arguments_fit={"verbose": 1},
        arguments_extract={"verbose": 1, "histogram": {"bins": 5}})

    print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))"""