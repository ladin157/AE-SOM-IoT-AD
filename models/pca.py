from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score


def pca_process(X, num_features=None, method='pca'):
    if num_features is None:
        num_features = X.shape[1]
    else:
        num_features = min(X.shape[1], num_features)
    if method.__eq__('kernel'):
        clf = KernelPCA(n_components=num_features)
    elif method.__eq__('sparse'):
        clf = SparsePCA(n_components=num_features)
    elif method.__eq__('truncated'):
        clf = TruncatedSVD(n_components=num_features)
    elif method.__eq__('incremental'):
        clf = IncrementalPCA(n_components=num_features)
    else:
        clf = PCA(n_components=num_features)
    clf.fit(X=X)
    X_transformed = clf.transform(X=X)
    return X_transformed, clf


def _testing(X_train, y_train, X_test, y_test, method='pca'):
    print("Using method {}".format(method))
    clf, X_train = pca_process(X=X_train, num_features=2, method=method)
    X_test = clf.transform(X_test)  # pca_process(X=X_test, num_features=2, method=method)
    # print(data_pca[:10])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=2)
    # auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    # print("AUC score: ", auc_score)

# if __name__ == '__main__':
#     data, target = load_iris(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1, shuffle=True)
#     # method PCA
#     print("Method PCA")
#     _testing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='pca')
#     print("Method Kernel PCA")
#     _testing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='kernel')
#     print("Method Sparse PCA")
#     _testing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='sparse')
#     print("Method Truncated PCA")
#     _testing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='truncated')
#     print("Method Incremental PCA")
#     _testing(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='incremental')
