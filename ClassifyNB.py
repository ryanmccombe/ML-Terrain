from sklearn.naive_bayes import GaussianNB


def classify(features_train, labels_train):
    return GaussianNB().fit(features_train, labels_train)
