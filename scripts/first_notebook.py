#%%
from scripts.utilities import *
filename = "./data/germeval2018.training.txt"
X_train, Y_train1, Y_train2 = get_train_data(filename)
filename = "./data/germeval2018.test.txt"
X_test, Y_test1, Y_test2 = get_train_data(filename)

#%%
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

#%%
count_vect.vocabulary_.get('ich')
#%%
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#%%
Y_train1[:10]

#%%
clf = MultinomialNB().fit(X_train_tfidf, Y_train1)
#%%
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(X_train, Y_train1)  
#%%
predicted = text_clf.predict(X_test)
np.mean(predicted == Y_test1)      

#%%
from collections import Counter
Counter(Y_test1)

#%%
2330/(2330+1202)

#%%
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(X_train_tfidf, Y_train1)  

predicted = text_clf.predict(X_test)
np.mean(predicted == Y_test1)            
#%%
