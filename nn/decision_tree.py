import math
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def entropy(y):
    """计算熵"""
    counts = Counter(y)
    res = 0.0
    for num in counts.values():
        p = num / len(y)
        res += -p * math.log2(p)  # 香农熵 E = - p * log(p)用来表示一个数据集的混乱度，在（0，1）内越小数据集越准确
    return res


def split_dataset(X, y, feature_index, threshold):
    """根据特征和阈值划分数据集"""
    left_X, left_y = [], []
    right_X, right_y = [], []

    for i, sample in enumerate(X):
        if sample[feature_index] <= threshold:
            left_X.append(sample)
            left_y.append(y[i])
        else:
            right_X.append(sample)
            right_y.append(y[i])

    return (tuple(left_X), tuple(left_y)), (tuple(right_X), tuple(right_y))


class Node:

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 用于划分的特征索引
        self.threshold = threshold  # 划分的阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的类别值


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = 0  # 对熵是最大化

        num_features = len(X[0])
        # 遍历每个feature中的每一个value，按照这个阈值进行分类，算出香农熵，并且使其最大化
        for feature_index in range(num_features):
            thresholds = Counter([row[feature_index] for row in X]).keys()
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                score = (len(y_left) / len(y)) * entropy(y_left) + (len(y_right) / len(y)) * entropy(y_right)

                if score > best_score:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_score = score

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(Counter(y)) == 1:
            return Node(value=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)

        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_index, threshold)
        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature_index=feature_index, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, node, x):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        res = []
        for x in X:
            tmp = self._predict_one(self.tree, x)
            res.append(tmp)
        return res


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 使用自己的决策树构造
    clf = DecisionTree(max_depth=50)
    clf.fit(X_train, y_train)
    # 进行预测
    y_pred = clf.predict(X_test)
    # 评估模型
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 使用sklearn内置决策树算法
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)  # 也可以使用'gini'
    # 训练决策树
    clf.fit(X_train, y_train)
    # 进行预测
    y_pred = clf.predict(X_test)
    # 评估模型
    print("Accuracy:", accuracy_score(y_test, y_pred))
