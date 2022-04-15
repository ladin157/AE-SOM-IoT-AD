from sklearn.preprocessing import MaxAbsScaler


class Scaling:
    def __init__(self):
        self.scaler = MaxAbsScaler()

    def fit(self, train_data):
        self.scaler.fit(train_data)

    def transform(self, test_data):
        transformed_data = self.scaler.transform(test_data)
        return transformed_data

