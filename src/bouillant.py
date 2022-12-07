
"""datas            print(runtime)et = openml.datasets.get_dataset(dataset_id=179)
print("f11", sklearn.metrics.f1_score(pred, y_test[dataset]))X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
print("zeit", suchzeit)                                                                             target=dataset.default_target_attribute)
print(runtime)
def class_occurences( X, y):
if len(y.shape) == 2:
occurences = []
for i in range(y.shape[1]):
   occurences.append(class_occurences(X, y[:, i]))
return occurences
else:
occurence_dict = defaultdict(float)
for value in y:
   occurence_dict[value] += 1
return occurence_dict


test=class_occurences(X,y)
print(type(test))
print(test)
"""






"""
def MissingValues(X): #ich gehe davon aus das wird nicht gebraucht für unified data representation
    missing = pd.isna(X)
    return missing
test=MissingValues(X)
print(test.shape)

def NumberOfMissingValues( X):
    if scipy.sparse.issparse(X):
        return float(MissingValues(X).sum())
    else:
        return float(np.count_nonzero(MissingValues(X)))
test=NumberOfMissingValues(X)
print(test)

print(X.shape)


def LogNumberOfFeatures( X):
    return np.log(get_number_features(X))

def LogNumberOfInstances( X):
        return np.log(get_number_instance(X))

test=LogNumberOfFeatures(X)
print(test)
test=LogNumberOfInstances(X)
print(test)
def class_occurences( X, y):
    if len(y.shape) == 2:
        occurences = []
        for i in range(y.shape[1]):
            occurences.append(class_occurences(X, y[:, i]))
        return occurences
    else:
        occurence_dict = defaultdict(float)
        for value in y:
            occurence_dict[value] += 1
        return occurence_dict


test=class_occurences(X,y)
print(test)
print("ttype",type(test))

def ClassProbabilitySTD(X, y):
    occurence_dict = class_occurences(X,y) # kann class_occurences von hier extraiert werden

    if len(y.shape) == 2:
        stds = []
        for i in range(y.shape[1]):
            std = np.array(
                [occurrence for occurrence in occurence_dict[i].values()],
                dtype=np.float64
            )
            std = (std / y.shape[0]).std()
            stds.append(std)
        return np.mean(stds)
    else:
        occurences = np.array(
            [occurrence for occurrence in occurence_dict.values()], dtype=np.float64
        )
        return (occurences / y.shape[0]).std()





def ClassProbabilityMean(X, y):
    occurence_dict = class_occurences(X,y)

    if len(y.shape) == 2:
        occurences = []
        for i in range(y.shape[1]):
            occurences.extend(
            [occurrence for occurrence in occurence_dict[i].values()]
            )
        occurences = np.array(occurences)
    else:
        occurences = np.array(
            [occurrence for occurrence in occurence_dict.values()], dtype=np.float64
        )
    return (occurences / y.shape[0]).mean()
test=ClassProbabilityMean(X,y)
print(test)


def ClassProbabilityMax(X, y):
    occurences = class_occurences(X,y)
    max_value = -1

    if len(y.shape) == 2:
        for i in range(y.shape[1]):
            for num_occurences in occurences[i].values():
                if num_occurences > max_value:
                    max_value = num_occurences
    else:
        for num_occurences in occurences.values():
            if num_occurences > max_value:
                max_value = num_occurences
    return float(max_value) / float(y.shape[0])

test=ClassProbabilityMax(X,y)
print(test)


def ClassProbabilityMin(X, y):
    occurences = class_occurences(X,y)

    min_value = np.iinfo(np.int64).max
    if len(y.shape) == 2:
        for i in range(y.shape[1]):
            for num_occurences in occurences[i].values():
                if num_occurences < min_value:
                    min_value = num_occurences
    else:
        for num_occurences in occurences.values():
            if num_occurences < min_value:
                min_value = num_occurences
    return float(min_value) / float(y.shape[0])
test=ClassProbabilityMin(X,y)
print(test)


def InverseDatasetRatio( X):
    return float(get_number_instance(X)) / float(
        get_number_features(X)
    )

test=InverseDatasetRatio(X)
print(test)


def LogInverseDatasetRatio(X):
    return np.log(InverseDatasetRatio( X))
test=LogInverseDatasetRatio(X)
print("looogg",test)


def DatasetRatio( X):
    return float(get_number_features(X)) / float(
        get_number_instance(X))

test=DatasetRatio(X)
print(test)
def LogDatasetRatio(X):
    return np.log(DatasetRatio( X))
test=LogDatasetRatio(X)
print("looogg",test)

def NumberOfCategoricalFeatures( categorical_indicator):
    return np.sum([value == True for value in categorical_indicator])
test=NumberOfCategoricalFeatures(categorical_indicator)
print(test)

def NumberOfNumericFeatures( categorical_indicator):
    return np.sum([value == False for value in categorical_indicator])
test=NumberOfNumericFeatures(categorical_indicator)
print(test)

def RatioNominalToNumerical(categorical_indicator):
    num_categorical = float(
        NumberOfCategoricalFeatures( categorical_indicator)
    )
    num_numerical = float(
        NumberOfNumericFeatures( categorical_indicator)
    )
    if num_numerical == 0.0:
        return 0.0
    else:
        return num_categorical / num_numerical
test = RatioNominalToNumerical(categorical_indicator)
print(test)

def RatioNumericalToNominal(categorical_indicator):
    num_categorical = float(
        NumberOfCategoricalFeatures( categorical_indicator)
    )
    num_numerical = float(
        NumberOfNumericFeatures( categorical_indicator)
    )
    if num_categorical == 0.0:
        return 0.0
    else:
        return num_numerical / num_categorical
test = RatioNumericalToNominal(categorical_indicator)
print(test)"""
"""print(X.shape)
preproceser = transformer(categorical_indicator)

X_transformer = preproceser.fit_transform(X,y)
print(type(X_transformer))
print(X_transformer.shape)
housi=csr_matrix(X_transformer).toarray()
print(housi)"""


"""rint(csr_matrix(X_transformer).toarray().shape)
b=metafeatures.NumSymbols(csr_matrix(X_transformer).toarray(), y,logging)"""
"""print(X.shape)
ilyes = metafeatures.MetafeatureFunctions()
s=metafeatures.SymbolsMax(X,y,ilyes)
print("ssss",s)"""

# metafeatures.ClassEntropy(X, y,logging,)
# metafeatures.SymbolsSum  #problem
# metafeatures.SymbolsSTD #problem
# metafeatures.SymbolsMean # problem
# metafeatures.SymbolsMax #problem
# metafeatures.SymbolsMin #problem
# metafeature.ClassOccurences
# metafeatures.ClassProbabilitySTD #problem
# metafeatures.ClassProbabilityMean #problem
# metafeatures.ClassProbabilityMax #problem
# metafeatures.ClassProbabilityMin #problem
# metafeatures.InverseDatasetRatio
# metafeatures.DatasetRatio
# metafeatures.RatioNominalToNumerical
# metafeatures.RatioNumericalToNominal
# metafeatures.NumberOfCategoricalFeatures
# metafeatures.NumberOfNumericFeatures
# metafeatures.MissingValues # problem
# metafeatures.NumberOfMissingValues #problem
# metafeatures.NumberOfFeaturesWithMissingValues #problem
# metafeatures.NumberOfInstancesWithMissingValues #problem
# metafeatures.NumberOfFeatures # valid print(a.value)
"""a= metafeatures.NumberOfFeatures  (X, y, logging_,feat_type="METAFEATURE")
print(a)"""





"""
mfe = MFE(features=["ClassEntropy", "NumSymbols", "SymbolsSum"])"""
"""
mfe = MFE(features=["ClassEntropy", "NumSymbols", "SymbolsSum", "SymbolsSTD", "SymbolsMean", "SymbolsMax", "SymbolsMin", "ClassOccurences", "ClassProbabilitySTD", "ClassProbabilityMean", "ClassProbabilityMax", "ClassProbabilityMin", "InverseDatasetRatio", "DatasetRatio", "RatioNominalToNumerical", "RatioNumericalToNominal", "NumberOfCategoricalFeatures", "NumberOfNumericFeatures", 'MissingValues', 'NumberOfMissingValues', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues', 'NumberOfFeatures'
'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio', 'LogDatasetRatio', 'PercentageOfMissingValues', 'PercentageOfFeaturesWithMissingValues', 'PercentageOfInstancesWithMissingValues', 'LogNumberOfFeatures', 'LogNumberOfInstances'
'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio', 'LogDatasetRatio', 'PercentageOfMissingValues', 'PercentageOfFeaturesWithMissingValues', 'PercentageOfInstancesWithMissingValues', 'LogNumberOfFeatures', 'LogNumberOfInstances'])
mfe.fit(X,y)"""
"""mfe.fit(X,y)
ft = mfe.extract()
print(ft)"""

"""best_feature = np.zeros((X.shape[0],3))
for index_feature in range (3):
    best_feature[:, index_feature]=X[:, sort_idx[index_feature]].T
print(best_feature)
print(best_feature.shape)"""






"""mfe = MFE(groups=["Statistical"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
print(type(ft[1][0]))
for i in range(len(ft[1])):
    if np.isnan(ft[1][i]):
        print("3asba")
    else:
        print(f"{ft[0][i]}#####{ft[1][i]}")
"""

"""ft = meta_features_model(X_transformer, y)
for a, b in zip(ft[0], ft[1]):
    X_unified = np.insert(X_unified, X_unified.shape[1], values=b, axis=1)
    column_name.append(f"{a}")"""


"""ft= meta_features_model(X_transformer,y)
    for a, b in zip(ft[0], ft[1]):
        X_unified = np.insert(X_unified, X_unified.shape[1], values=b, axis=1)
        column_name.append(f"{a}")
        print("ccccccccccccccccc",a, "hhhhhhhhhhhh", b)"""