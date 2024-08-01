class DecisionTree:
    def _init_(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1:
            return {'emotion': y[0]}

        if self.max_depth and depth >= self.max_depth:
            return {'emotion': max(y, key=y.count)}

        best_split = self.get_best_split(X, y)
        if best_split is None:
            return {'emotion': max(y, key=y.count)}

        left_idx, right_idx, split_feature, split_value = best_split
        node = {'split_feature': split_feature, 'split_value': split_value}
        node['left'] = self.fit(X[left_idx], y[left_idx], depth + 1)
        node['right'] = self.fit(X[right_idx], y[right_idx], depth + 1)
        return node

    def get_best_split(self, X, y):
        best_gini = 1
        best_split = None
        for feature in range(len(X[0])):
            for value in set(X[:, feature]):
                left_idx = [i for i in range(len(X)) if X[i, feature] <= value]
                right_idx = [i for i in range(len(X)) if X[i, feature] > value]
                gini = self.calculate_gini(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (left_idx, right_idx, feature, value)
        return best_split

    def calculate_gini(self, left_y, right_y):
        total_samples = len(left_y) + len(right_y)
        gini_left = 1 - sum([(left_y.count(label) / len(left_y)) ** 2 for label in set(left_y)])
        gini_right = 1 - sum([(right_y.count(label) / len(right_y)) ** 2 for label in set(right_y)])
        gini = (len(left_y) / total_samples) * gini_left + (len(right_y) / total_samples) * gini_right
        return gini

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while 'emotion' not in node:
                if sample[node['split_feature']] <= node['split_value']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['emotion'])
        return predictions

# Example data
X_train = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
y_train = ['happy', 'happy', 'sad', 'sad', 'hungry']

# Dummy encoding for simplicity
X_train = np.array(X_train)
y_train_encoded = [1 if label == 'happy' else 0 for label in y_train]

# Train the decision tree model
model = DecisionTree(max_depth=2)
model.tree = model.fit(X_train, y_train_encoded)

# Example prediction
X_test = [[1, 2], [4, 3]]
predictions = model.predict(X_test)
print(predictions)