from sklearn.preprocessing import StandardScaler, MaxAbsScaler, normalize


def standardize_data(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    return scaler, X_train_scaled


def scale_data(X_train, X_test=None):
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None
    return scaler, X_train_scaled, X_test_scaled


def normalize_data(X_train, X_test=None):
    X_train_normalized, X_test_normalized = None, None
    if X_train is not None:
        X_train_normalized = normalize(X_train)
    if X_test is not None:
        X_test_normalized = normalize(X_test)
    return X_train_normalized, X_test_normalized
