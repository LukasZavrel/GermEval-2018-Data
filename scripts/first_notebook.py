#%%
from scripts.utilities import *
filename = "./data/germeval2018.training.txt"
X_train, Y_train1, Y_train2 = get_train_data(filename)
filename = "./data/germeval2018.test.txt"
X_test, Y_test1, Y_test2 = get_train_data(filename)

#%%
count_vect = CountVectorizer(min_df=1)
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
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1,3)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1, scoring=make_scorer(f1_score, average='macro'))
gs_clf = gs_clf.fit(X_train, Y_train1)                              
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
#%%
gs_clf.best_params_
#%%
gs_clf.best_score_    
#%%
from collections import Counter
Counter(Y_test1)
print("Default score: ",2330/(2330+1202)

#%%
text_clf.set_params(**gs_clf.best_params_)
text_clf.fit(X_train, Y_train1)  
#%%
predicted = text_clf.predict(X_test)
np.mean(predicted == Y_test1)    
#%%  
from sklearn.metrics import f1_score
f1_score(predicted, Y_test1, average='macro')  