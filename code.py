
import pandas as pd
import numpy as np
from stop_words import get_stop_words
import collections
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.pyplot as plt


data_yt = pd.read_csv("/home/gnu/github/analisis_comments_youtube/data_wrangling_yt.csv")
data_yt.info()

# Filtro ascending por likes en comentarios
df_comments_data = data_yt.iloc[:,7:]
likes_comments_filter = df_comments_data.sort_values('likes_comment',ascending=False)
likes_comments_filter.describe()


###################################################################################
                               **** TOP 10 WORDS ****
comments_label = df_comments_data["comment_text"]
comments_label = list(comments_label)

## Cleaning lyrics songs
def comments_cleaning():
    comment = comments_label
    comment = str(comment)
    comment = comment.lower()

    comment = comment.replace("\\n",'')
    comment = comment.replace('"','')
    comment = comment.replace(',','')
    comment = comment.replace(':','')
    comment = comment.replace('¿','')
    comment = comment.replace('?','')
    comment = comment.replace('(','')
    comment = comment.replace(')','')
    comment = comment.replace('!','')
    comment = comment.replace('¡','')
    comment = comment.replace('.','')
    comment = comment.replace("'",'')
    return comment
    
comments = comments_cleaning()
comments

# Despreciar palabras
stop_words_en = get_stop_words('en') # English StopWords    
stop_words_es = get_stop_words('es') # Spanish StopWords
stopWords = stop_words_es + stop_words_en
stopWords +=  ["bien","siempre","volante","video","escorpion","dorado","ahora","noroña","asi","buen","", "jajajaja", "jajaja", "sigue", "si", "xd", "vez", "veo", "aquí","hace", "q", "ser" , "tan"] #Adding aditional StopWords  

def top_10_words():
    filtered_words = [word for word in comments.split() if word not in stopWords]
    counted_words = collections.Counter(filtered_words)

    words = []
    counts = []
    for letter, count in counted_words.most_common(20):
        words.append(letter)
        counts.append(count)

    i = 1
    for palabras in words:
        print(i, palabras)
        i+=1
    
    # Graphic list
    colors = cm.rainbow(np.linspace(0, 1, 10))
    rcParams['figure.figsize'] = 10, 5

    plt.title('Top words in the text vs their count')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.barh(words, counts, color="blue")

top_10_words()


###################################################################################3

from googletrans import Translator

# Language processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize

                  ****ANALISIS DE SENTIMIENTOS EN COMENTARIOS****
nltk.download('vader_lexicon')
nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
analizador = SentimentIntensityAnalyzer()
translator = Translator()

comments_label = df_comments_data["comment_text"]
comments_label = list(comments_label)

cont = 0
translate = []
scores_comments =[]
qualifys = []
for row in comments_label:
    cont += 1
    print(row)
    # Translate es_en
    translate_row = translator.translate(row, src="es", dest="en")
    row_en = translate_row.text
    translate.append(row_en)
    print(row_en)

    # Analisis emotion
    scores = analizador.polarity_scores(row_en)
    if scores["compound"] > 0:
        score = scores["compound"]
        scores_comments.append(score)
        qualify = "positive"
        qualifys.append(qualify)
        print(score , qualify)
        print('###########################################')
    elif scores["compound"] < 0:
        score = scores["compound"]
        scores_comments.append(score)
        qualify = "negative"
        qualifys.append(qualify)
        print(score, qualify)
        print('###########################################')
    elif scores["compound"] == 0:
        score = scores["compound"]
        scores_comments.append(score)
        qualify = "neutral"
        qualifys.append(qualify)
        print(score, qualify)
        print('###########################################')




dictt = {"comments_es": df_comments_data["comment_text"],
         "comments_en": translate,
         "likes": df_comments_data["likes_comment"],
         "scores": scores_comments,
         "qualify": qualifys}



# Filtros de comentarios negativos 
df = pd.DataFrame(dictt)
df
# Comentarios Negativos
filtro_negative = df["qualify"] == "negative"
df_negative = df[filtro_negative]
df_negative = df_negative.sort_values('likes',ascending=False)
df_negative = df_negative.sort_values('scores',ascending=True)
df_negative.count()

# Filtros de comentarios positivos
filtro_positive = df["qualify"] == "positive"
df_positive = df[filtro_positive]
df_positive = df_positive.sort_values('likes',ascending=False)
df_positive = df_positive.sort_values('scores',ascending=False)
df_positive.count()

# Filtros de comentarios neutral
filtro_neutral = df["qualify"] == "neutral"
df_neutral = df[filtro_neutral]
df_neutral = df_neutral.sort_values('likes',ascending=False)
df_neutral.count()


FIN
