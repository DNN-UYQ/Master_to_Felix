"""ft = get_meta_features(X_transformer, y)
    for name, meta_feature in zip(ft[0], ft[1]):
        column_name.append(name)
        X_new = np.insert(X_new, X_new.shape[1], values=meta_feature, axis=1)"""
    #X_new=np.insert(X_new, X_new.shape[1], values=get_number_features(X_transformer), axis=1)

    #X_new=np.insert(X_new, X_new.shape[1], values=get_number_instance(X_transformer), axis=1)



#NumSymbols
def _calculate(X, y):
    categorical = {
        key: True if value.lower() == "categorical" else False
        for key, value in feat_type.items()
    }
    symbols_per_column = []
    for i in range(X.shape[1]):
        if categorical[X.columns[i] if hasattr(X, "columns") else i]:
            column = X.iloc[:, i] if hasattr(X, "iloc") else X[:, i]
            unique_values = (
                column.unique() if hasattr(column, "unique") else np.unique(column)
            )
            num_unique = np.sum(pd.notna(unique_values))
            symbols_per_column.append(num_unique)
    return symbols_per_column


#mETAFreature
class SymbolsSTD(MetaFeature):
    def _calculate(self, X, y, logger, feat_type):
        values = [val for val in helper_functions.get_value("NumSymbols") if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0

#SymbolsMean
class SymbolsMean(MetaFeature):
    def _calculate(self, X, y, logger, feat_type):
        # TODO: categorical attributes without a symbol don't count towards this
        # measure
        values = [val for val in helper_functions.get_value("NumSymbols") if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0

#SymbolsMax
@metafeatures.define("SymbolsMax", dependency="NumSymbols")
class SymbolsMax(MetaFeature):
    def _calculate(self, X, y, logger, feat_type):
        values = helper_functions.get_value("NumSymbols")
        if len(values) == 0:
            return 0
        return max(max(values), 0)



#Symbolmin
@metafeatures.define("SymbolsMin", dependency="NumSymbols")
class SymbolsMin(MetaFeature):
    def _calculate(self, X, y, logger, feat_type):
        # The minimum can only be zero if there are no nominal features,
        # otherwise it is at least one
        # TODO: shouldn't this rather be two?
        minimum = None
        for unique in helper_functions.get_value("NumSymbols"):
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        return minimum if minimum is not None else 0









def missing_values(X):
    missing = pd.isna(X)
    return missing


def number_Features_mv(X):
    missing =missing_values(X)
    num_missing = missing.sum(axis=0)
    return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


def number_instances_mv(X):
    missing = missing_values(X)
    num_missing = missing.sum(axis=1)
    return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


def percentage_instances_mv(X):
    n_missing = number_instances_mv(X)
    n_total = get_number_instance(X)
    return float(n_missing / n_total)


def percentage_features_mv(X):
    n_missing = number_Features_mv(X)
    n_total = get_number_features(X)
    return float(n_missing / n_total)

def number_missing_values(X):
    missing = np.count_nonzero(np.isnan(X))
    return missing
test=number_missing_values(X)
print(test)


def PercentageOfMissingValues(X):
    return float(number_missing_values(X)) / float(
        X.shape[0] * X.shape[1]
    )
test=PercentageOfMissingValues(X)
print(test)

