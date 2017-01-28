#python 3
import string
import math
import numpy as np
from collections import Counter
import re
from sklearn import preprocessing
readFile = open("news1.txt","r") #create file object for reading file
mydoc = readFile.read() #return string from file
#print(mydoc+'\n\n')
#remove everything except \n\s\w
mydoc =re.sub(r'[^\w\s?\n]','',mydoc)
mydoclist = mydoc.split('\n') #make list 
mydoclist = [x for x in mydoclist if x != '']
print(mydoclist)
print('\n\n')
print('Number of document are : ' + str(mydoclist.__len__()) + '\n')
countdoc=1
#print individual items of list
for x in mydoclist:
    print('Document number is : ' + str(countdoc) + '\n')
    print(x + '\n')
    countdoc+=1
#count frequency of each word in list
countdoc=1
for doc in mydoclist:
    tf=Counter()
    for word in doc.split():
        tf[word]+=1
    print('tf for document : ' + str(countdoc) + '\n')
    print(tf.items())
    countdoc+=1
    print('\n')
print('\n\n')
#make term frequency vector
def build_lexicon(data):
        lexicon = set()
        for doc in data:
            lexicon.update([word for word in doc.split()])
        return lexicon

#count tf
def tf(term, document):
    return freq(term, document)
def freq(term, document):
    return document.split().count(term)

#my word list
myWordList = build_lexicon(mydoclist)
print('\n\n')
doc_term_matrix = []
#The method join() returns a string in which the string elements of sequence have been joined by ','
print("Our wordlist is :  [" + ','.join(list(myWordList)) + "]")
print('\n\n')

for doc in mydoclist:
    print('The doc is: "' + doc + '"')
    tf_vector = [tf(word, doc) for word in myWordList]
    tf_vector_string = ','.join(format(freq,'d') for freq in tf_vector)
    print ('The tf vector for Document : %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string))
    print('\n')
    doc_term_matrix.append(tf_vector)
print ('\n\nAll combined, here is our master document term matrix: ')
print (doc_term_matrix)
#A regular old document matrix
print('\n\n')
print("Correspoiding Matrix is:")
np.set_printoptions(threshold=np.nan)
print(np.matrix(doc_term_matrix))
#normalize vectors(matrix)
doc_term_matrix_l2 = preprocessing.normalize(doc_term_matrix, norm='l2')
print(np.matrix(doc_term_matrix_l2))
print('\n\n')

def NumDocContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 1:
           doccount += 1
    return doccount
#function for idf
def idf(word, doclist):
           n_samples = len(doclist)
           df = NumDocContaining(word, doclist)
           return np.log(n_samples /1+df)

my_idf_vector = [idf(word, mydoclist) for word in myWordList]
print ('Our vocabulary vector is [' + ', '.join(list(myWordList)) + ']')
print ('The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']')
#build idf matrix
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

my_idf_matrix = build_idf_matrix(my_idf_vector)
doc_term_matrix_tfidf = []
#performing tf-idf matrix multiplication
for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))
doc_term_matrix_tfidf_l2=preprocessing.normalize(doc_term_matrix_tfidf, norm='l2')

print('\n\n')
print(myWordList)
print('\n\n')
print(np.matrix(doc_term_matrix_tfidf_l2))
