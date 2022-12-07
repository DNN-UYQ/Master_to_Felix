import sklearn.tree
from sklearn.datasets import load_iris
from pymfe.mfe import MFE
from Clf_list import classfier
from Datasets_info import get_itemset, get_statistical
MFE.metafeature_description()

clf_list=classfier()
iris = load_iris()
X=iris.data
Y=iris.target
mfe = MFE(features=[ "eq_num_attr" ])
mfe.fit(X, Y)
ft = mfe.extract()
meta=[]
name=[]
for x, y in zip(ft[0], ft[1]):
    name.append(x)
    meta.append(y)
print(len(name))
print("meta",meta)
print("name",name)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

MFE.metafeature_description()

ft= get_statistical(X, Y)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

"""mfe = MFE(groups=["itemset"])
mfe.fit(X, Y)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
mfe.fit(X)
ft = mfe.extract()
print(ft)"""











for i in range(len(clf_list)):
    # Extract from model

    model = clf_list[i]
    x=iris.data
    y=iris.target

    model.fit(x, y)

    extractor = MFE(features=["var_importance"])
    ftft = extractor.extract_from_model(model)
    print("fttttttttttttttttttttttttttttttttt",ftft)
    print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ftft[0], ftft[1])))
    # Extract specific metafeatures from model"""
