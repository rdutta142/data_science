
# coding: utf-8

# In[ ]:

def classify(features_train,labels_train)
    #import sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    #create classifier
    clf = GaussianNB()
    #fit the classifier on the training features and labels
    r = clf.fit(features_train,labels_train)
    #return the fit classifier
    return r

