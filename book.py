import string
import random
import numpy as np
import os
import re
import nltk
import pickle
from afinn import Afinn
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
nltk.download('stopwords')
# Get a list of stopwords from nltk
stopwords = nltk.corpus.stopwords.words("english")

def get_clean_words(book_filename, category):
    """takes in the name of the book file and wether it is like or dislike
        returns a list of every word used excluding stopwords"""
    def _isnum(w):
        try:
            int(w)
            return True
        except ValueError:
            return False

    # Load her markup
    with open("text/%s/%s" % (category, book_filename)) as fp:
        markup = fp.read()
    # Remove table and external links
    markup_text = re.sub(r'\{\{[\s\S]*?\}\}', '', markup)
    # Remove category links
    markup_text = re.sub(r'\[\[Category.+\]\]', '', markup_text)
    # Set words to lowercase and remove them if they are stop words
    words = [w.lower() for w in re.findall('\w+', markup_text) if w.lower() not in stopwords]
    # Remove numbers
    words = [w for w in words if not _isnum(w)]
    return words

                    ##SENTENCE LENGTH##


                        ##SENTIMENT ANALYSIS##

def text_string_seperate(category):
    bookList = []
    book_name_list = []
    #looks at a character in a faction
    for file_name in os.listdir("text/%s" % category):
        book_list = []
        book_name_list.append(file_name)
        #looks at each word for that character
        for word in (get_clean_words(file_name, category)):
            #creates a list of all words in that characters artical
            book_list.append(word)
        #combines all words into a string
        book_string = ' '.join(book_list)
        bookList.append(book_string)
    return [bookList,book_name_list]

def sentiment_list(category):
    afinn = Afinn()
    #genrates list of strings of all the words in a characters article
    bookList = text_string_seperate(category)
    sentiment = []
    for i in range(len(bookList[0])):
        #creates a list of the sentiment scores of all characters in the faction
        wordCount = len(get_clean_words(bookList[1][i],category))
        sentiment.append([afinn.score(bookList[0][i])/wordCount,bookList[1][i]])
    sentiment.sort()
    sentiment_score = []
    sentiment_name = []
    for book in sentiment:
        sentiment_score.append(book[0])
        sentiment_name.append(book[1])
    return [sentiment_score,sentiment_name]

def sentiment_plot():
    [sentiment_like_y, sentiment_like_x] = sentiment_list("like")
    [sentiment_dislike_y, sentiment_dislike_x] = sentiment_list("dislike")

    #plt.figure(figsize = (20,5))
    plt.subplot(2,1,1)
    plt.ylim([-.05,.12])
    xs = [i + .01 for i, _ in enumerate(sentiment_like_x)]

    plt.bar(xs,sentiment_like_y, color = 'red', alpha = .5)
    plt.title('Sentiment Scores of Like Books', fontsize = 20)
    #plt.xticks([i for i, _ in enumerate(sentiment_like_x)],sentiment_like_x,fontsize = 10, rotation = 'vertical')
    plt.yticks(fontsize = 20)

    plt.ylabel('Name of Liked Book', fontsize = 20)

    plt.subplot(2,1,2)
    plt.ylim([-.05,.12])
    xs = [i + .01 for i, _ in enumerate(sentiment_dislike_x)]

    plt.bar(xs,sentiment_dislike_y, color = 'blue', alpha = .5)
    plt.title('Sentiment Scores of Dislike Books', fontsize = 20)
    #plt.xticks([i for i, _ in enumerate(sentiment_dislike_x)],sentiment_dislike_x,fontsize = 10, rotation = 'vertical')
    plt.yticks(fontsize = 20)

    plt.ylabel('Name of Disliked Book', fontsize = 20)
    plt.show()

                            ##BAG OF WORDS MATRIX##

def text_string(category):
    factionList = []
    for file_name in os.listdir("text/%s" % category):
        for word in (get_clean_words(file_name, category)):
            factionList.append(word)
    return factionList

def text_string_total():
    totalText = []
    for category in ["like", "dislike"]:
        for word in (text_string(category)):
            if word in totalText:
                continue
            else:
                totalText.append(word)
    return totalText

def create_bow_matrix():
    wordList = text_string_total()
    a = [t[::-1] for t in enumerate(wordList)]
    wordDict  = dict(a)
    bow_matrix = []
    for category in ["like", "dislike"]:
        for file_name in os.listdir("text/%s" % category):
            char_words = (np.zeros(len(wordList)))
            for word in (get_clean_words(file_name, category)):
                if word in wordDict:
                    char_words[wordDict[word]] += 1
            bow_matrix.append(char_words)
    return bow_matrix

    ###PCA ####

def pca(bow_matrix,number_of_like):
    pca = PCA(n_components=10)
    pca.fit(bow_matrix)
    like_dimension1 = np.matmul(bow_matrix[:(number_of_like-1)],pca.components_[2])
    like_dimension2 = np.matmul(bow_matrix[:(number_of_like-1)],pca.components_[1])
    dislike_dimension1 = np.matmul(bow_matrix[(number_of_like-1):],pca.components_[2])
    dislike_dimension2 = np.matmul(bow_matrix[(number_of_like-1):],pca.components_[1])
    return [(like_dimension1,like_dimension2),(dislike_dimension1,dislike_dimension2)]

            ##TSNE##

            #PCA SCATTERPLOT##
def pca_plot():
    (like_dimension1,like_dimension2) = pca(bow_matrix,13)[0]
    (dislike_dimension1,dislike_dimension2) = pca(bow_matrix,13)[1]
    #creates scatterplot
    plt.scatter(np.log(like_dimension1),np.log(like_dimension2),80,'red',label = 'Liked Book')
    plt.scatter(np.log(dislike_dimension1),np.log(dislike_dimension2),80,'blue',label = 'Disliked Book')

    #adds title
    plt.title('PCA Plot', fontsize = 25)
    #adds axis
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    plt.xlabel('Log 10 of Dimension 1', fontsize = 20)
    plt.ylabel('Log 10 of Dimension 2', fontsize = 20)

    plt.legend(loc=9)
    plt.show()

def tsne(bow_matrix,number_of_likes):
    X_tsne = TSNE(n_components=2, perplexity = 5).fit_transform(bow_matrix)
    likes_tsne1 = []
    likes_tsne2 = []
    dislikes_tsne1 = []
    dislikes_tsne2 = []
    for i in range(number_of_likes):
        likes_tsne1.append(X_tsne[i][0])
        likes_tsne2.append(X_tsne[i][1])
    for i in range(len(bow_matrix)-number_of_likes):
        dislikes_tsne1.append(X_tsne[number_of_likes+i][0])
        dislikes_tsne2.append(X_tsne[number_of_likes+i][1])

    plt.scatter(likes_tsne1,likes_tsne2,80,'red',label = 'Liked Book')
    plt.scatter(dislikes_tsne1,dislikes_tsne2,80,'blue',label = 'Disliked Book')
    #adds title
    plt.title('TSNE of Transposed BOW Matrix', fontsize = 25)
    #adds axis
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    plt.xlabel('TSNE Dimension 1', fontsize = 20)
    plt.ylabel('TSNE Dimension 2', fontsize = 20)

    plt.legend(loc=9)
    plt.show()

def create_target_array():
    target_array = []
    for category in ["like", "dislike"]:
        for file_name in os.listdir("text/%s" % category):
            if category =='like':
                target_array.append([file_name,1])
            else:
                target_array.append([file_name,0])
    return target_array

                        ###TF IDF###

def create_tf_matrix(bow_matrix):
    tf = []
    for book in bow_matrix:
        normalized = []
        total = sum(book)
        for word in book:
            normalized.append(word/total)
        tf.append(normalized)
    return tf

def create_idf_matrix(bow_matrix):
    #creates list of total number of times
    #a word is mentioned
    idf_totals = []
    for i in range(len(bow_matrix[0])):
        word_count = 0
        for book in bow_matrix:
            if book[i] != 0:
                word_count += 1
        idf_totals.append(word_count)
    idf = []
    for i in range(len(bow_matrix[0])):
        idf.append(np.log(len(bow_matrix)/(idf_totals[i])))
    return idf

def create_tf_idf_matrix(bow_matrix):
    tf = create_tf_matrix(bow_matrix)
    idf = create_idf_matrix(bow_matrix)
    tf_idf = np.zeros((len(tf),len(tf[0])))
    for i in range(len(tf)):
        for j in range(len(idf)):
            tf_idf[i][j] = tf[i][j]*idf[j]
    return tf_idf

def commonWords(book_index,tf_idf_matrix,wordList):
    word_map = []
    for i in range(len(wordList)):
        word_map.append([tf_idf_matrix[book_index][i],i])
    sorted_word_map = sorted(word_map)
    highWords = []
    for i in range(40):
        highWords.append(wordList[sorted_word_map[-i-1][1]])
    return highWords

def allCommonWords(target_array,tf_idf_matrix,word_list):
    like_common_words = []
    dislike_common_words = []
    for i in range(len(target_array)):
        if target_array[i][1] == 1:
            like_common_words.append([target_array[i][0],commonWords(i,tf_idf_matrix,word_list)])
        else:
            dislike_common_words.append([target_array[i][0],commonWords(i,tf_idf_matrix,word_list)])
    for book in like_common_words:
        print('LIKE BOOK:', book[0])
        for word in book[1]:
            print ('    ', word)
    for book in dislike_common_words:
        print('DISLIKE BOOK:', book[0])
        for word in book[1]:
            print ('    ', word)
    return [like_common_words,dislike_common_words]

def wordLength(words):
    """finds the average word length in a list of words"""
    length_list = []
    for word in words:
        length_list.append(len(word))
    average_length = sum(length_list)/len(length_list)
    return average_length

def commonWordLength(like_common_words,dislike_common_words):
    like_word_length = []
    for book in like_common_words:
        like_word_length.append(wordLength(book[1]))
    dislike_word_length =[]
    for book in dislike_common_words:
        dislike_word_length.append(wordLength(book[1]))
    #like plot
    plt.hist(like_word_length, alpha = .5)
    plt.title('Average Word Length', fontsize = 25)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 20)

    plt.xlabel('Word Length', fontsize = 20)
    plt.ylabel('Number of Books', fontsize = 20)

    #dislike plot
    plt.hist(dislike_word_length,color = 'red', alpha = .5)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 20)
    #plt.axis([-300,600,0,600])

    plt.show()


if __name__ == "__main__":
    #word_list = text_string_total()
    #pickle.dump(word_list,open('word_total.txt', 'wb'))
    word_list = pickle.load(open('word_total.txt', 'rb'))

    #bow = create_bow_matrix()
    #pickle.dump(bow,open('bow_matrix.txt', 'wb'))
    bow_matrix = pickle.load(open('bow_matrix.txt', 'rb'))

    #sentiment_plot()

                        ##tf idf analysis
    #
    # tf_idf_matrix = create_tf_idf_matrix(bow_matrix)
    # target_array = create_target_array()
    # x = allCommonWords(target_array, tf_idf_matrix,word_list)
    #
    # commonWordLength(x[0],x[1])

    pca_plot()

    #tsne(bow_matrix,13)
