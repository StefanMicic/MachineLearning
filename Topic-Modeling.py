import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump,load

npr = []
entries = os.listdir('20news/')
for entry in entries:
    e = os.listdir('20news/'+entry+'/')
    for en in e:
        opened = open('20news/'+entry+'/'+en, "r")
        npr.append(opened.read())
        opened.close()

cv = CountVectorizer(max_df = 0.9,min_df = 3,stop_words = 'english')

dtm = cv.fit_transform(npr)
dump(dtm,'dtm.joblib')

LDA = LatentDirichletAllocation(n_components = 20,random_state = 42).fit(dtm)
dump(LDA,'LDA.joblib')

print(len(cv.get_feature_names()))
print('LDA components: ',LDA.components_)

LDA = load('LDA.joblib')
for i,topic in enumerate(LDA.components_):
    print("THE TOP 10 WORDS FOR {}".format(i))
    print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])

print(LDA.transform(dtm)[0].argmax())
'''
#Second way
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
tfidf = TfidfVectorizer(max_df = 0.95,min_df = 2,stop_words ="english")
model = NMF(n_components = 20,random_state = 42)
model.fit(dtm)
'''
