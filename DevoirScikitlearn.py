from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import glob

name='Naive Bayes'

def main(name):
    if name not in ['SVM', 'Regression Logistique', 'Arbre de decision', 'Random Forest', 'Naive Bayes']:
        print("Ce nom n'est pas valable pour ce script\nVeuillez choisir parmi les noms de modèles suivant:\n- 'Arbre de decision'\n- 'SVM'\n- 'Regression Logistique'\n- 'Random Forest'\n- 'Naive Bayes'")
        return
    text_data, target = recup_data()
    data = tokenize(text_data)
    X_train, X_test, y_train, y_test = vectorize(data, target)
    if name == 'SVM':
        train_SVM(X_train, X_test, y_train, y_test)
    elif name == 'Regression Logistique':
        train_RL(X_train, X_test, y_train, y_test)
    elif name == 'Arbre de decision':
        train_DT(X_train, X_test, y_train, y_test)
    elif name == 'Random Forest':
        train_RF(X_train, X_test, y_train, y_test)
    elif name == 'Naive Bayes':
        train_NB(X_train, X_test, y_train, y_test)
        

def recup_data():
    '''
    Implémentation des listes data et target qui contiennent respectivement les textes des reviews et la classification positive ou négative

    Parameters
    ----------

    Returnspip install -U scikit-learn
    -------
    list, list
        2 liste contenants les données d'entrées
    '''

    #On déclare deux listes vide qu'on va remplir dans les boucles suivantes
    data=[]
    target=[]

    #On lit d'abord les reviews positives
    for file in glob.glob('data/imdb_smol/pos/*'):
        with open(file, 'r') as f:
            data.append(f.read())   #On ajoute le texte dans la liste data
            target.append(0)        #On ajoute la classification dans la liste target
    #Puis les reviews négatives
    for file in glob.glob('data/imdb_smol/neg/*'):
        with open(file, 'r') as f :
            data.append(f.read())   #Idem
            target.append(1)        #Idem

    return data, target

def vectorize(text_data, target): 
    #On déclare la façon dont on va traiter les données textuelle (ici avec tf-idf)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        stop_words='english'
    )
    data = vectorizer.fit_transform(text_data)

    return train_test_split(data, target, test_size=0.3)

def tokenize(text_data):
    import spacy
    import re
    #loading the english language small model of spacy
    nlp = spacy.load('en_core_web_sm')
    sw_spacy = nlp.Defaults.stop_words
    new_text_data=[]
    for texts in text_data:
        texts=texts.lower()
        texts=re.sub(r"[^\w']+(br)?[^\w']?|[0-9]+", r' ', texts)
        texts=texts.strip()
        doc = nlp(texts)
        words=[]
        for token in doc :
            word = token.lemma_
            if word.lower() not in sw_spacy and re.match('\w+', word):
                words.append(word.lower())
        new_text = " ".join(words)
        new_text=new_text
        new_text_data.append(new_text)
    return new_text_data

def train_SVM(X_train, X_test, y_train, y_test):
    #### Entraînement avec un SVM ####
    from sklearn.svm import LinearSVC, SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score

    clf = LinearSVC(dual=False)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('result for SVM classifier :')
    print(classification_report(y_test, y_pred))
    scores = cross_val_score(clf,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='r2')
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))

    #print('Optimisation des hyper-parametres du SVM :')
    #param_grid =  {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel':['rbf','linear']}
    #grid = GridSearchCV(SVC(), param_grid, cv = 5, scoring = 'accuracy')
    #estimator = grid.fit(data, target)
    #estimator.cv_results_
    #
    #df = pd.DataFrame(estimator.cv_results_)
    #df.sort_values('rank_test_score')
    #
    #from IPython.display import display
    #display(df.sort_values('rank_test_score'))

def train_RL(X_train, X_test, y_train, y_test):
    #### Entraînement en Regression Logistique ####
    from sklearn.linear_model import LogisticRegression
    # Train model
    lr = LogisticRegression().fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    
    
    print('results for Linear Regression :')
    print(classification_report(y_test, y_pred))
    
    #################################################
    #Accuracy = 88%
    
def train_DT(X_train, X_test, y_train, y_test):
    ##### Entraînement Arbre de décision ####
    from sklearn import tree
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    
    print('Results for Decision Tree :')
    print(classification_report(y_test, y_pred))
        ##########################################
    #Accuracy = 68%

def train_RF(X_train, X_test, y_train, y_test):
    #### Entraînement randdom forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=10)
    clf = clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    
    print('Results for Random Forest :')
    print(classification_report(y_test, y_pred))
        #####################################

def train_NB(X_train, X_test, y_train, y_test):
    #### Entrainement naive Bayes ####
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Results for Naive Bayes :')
    print(classification_report(y_test, y_pred))

    ##################################
    
main(name)