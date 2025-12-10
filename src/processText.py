import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.corpus import wordnet


# Helper functions
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # 'a'
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # 'v'
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # 'n'
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  # 'r'
    else:
        return wordnet.NOUN # Default to noun


#__TEXT_PREPROCESSING__
def processText(text):
    text = text.lower()
    text = re.sub(r"\s+" , " " ,text)
    text =re.sub(r"[^\w\s]", "", text)

    #Tokenizing the text
    tokens = word_tokenize(text)


    #Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]


    #Lemmatizing the text
    tagged_tokens = nltk.pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos = get_wordnet_pos(tag)) for word, tag in tagged_tokens]
    return ' '.join(lemmatized_tokens)

