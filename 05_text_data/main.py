import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
# from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import ssl

from bs4 import BeautifulSoup

EASY_HAM_1_PATH = '01_Processing/spam_assassin_corpus/easy_ham_1'
EASY_HAM_2_PATH = '01_Processing/spam_assassin_corpus/easy_ham_2'
SPAM_1_PATH = '01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = '01_Processing/spam_assassin_corpus/spam_2'

DATA_JSON_FILE = '01_Processing/email-text-data.json'

SPAM_CAT = 1
HAM_CAT = 0


stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english')


def download_nltk_resources():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('stopwords')


# download_nltk_resources()


def extract_email(file):
    with open(file=file, mode='r', encoding='latin-1') as message:
        lines = message.readlines()

    body = None

    try:
        body_index = lines.index('\n')

    except ValueError:
        pass

    else:
        body = lines[body_index:]

        for line in body:
            if line == '\n':
                body.remove(line)

        body = '\n'.join(line.strip() for line in body if line != '\n')

    finally:
        return body


# email_body = extract_email(file='01_Processing/practice_email.txt')
# print(email_body)

# Email Body Extraction


def email_body_generator(path):
    for root, dirnames, filenames in os.walk(path):
        for file_name in filenames:
            file_path = join(root, file_name)

            body = extract_email(file=file_path)
            yield file_name, body


def df_from_directory(path, classification):
    rows = []
    row_names = []
    for file_name, body in email_body_generator(path=path):
        rows.append({'MESSAGE': body, 'CLASSIFICATION': classification})
        row_names.append(file_name)

    return pd.DataFrame(rows, index=row_names)


def get_data():
    spam_emails = df_from_directory(path=SPAM_1_PATH, classification=SPAM_CAT)
    spam_emails = pd.concat([spam_emails, df_from_directory(path=SPAM_2_PATH, classification=SPAM_CAT)])
    # print(spam_emails.head())
    # print(spam_emails.shape)

    ham_emails = df_from_directory(path=EASY_HAM_1_PATH, classification=HAM_CAT)
    ham_emails = pd.concat([ham_emails, df_from_directory(path=EASY_HAM_2_PATH, classification=HAM_CAT)])
    # print(ham_emails.head())
    # print(ham_emails.shape)

    df = pd.concat([spam_emails, ham_emails])
    # print(df.shape)
    # print(df.head())
    # print(df.tail())

    # Check null
    # print(df['MESSAGE'].isnull().values.any())
    # print(df[df.MESSAGE.isnull()].index)
    # print(df.index.get_loc('.DS_Store'))
    #
    # print(df[692:695])

    df = df.drop(['.DS_Store'])
    # print(df['MESSAGE'].isnull().values.any())
    # print(df[df.MESSAGE.isnull()].index)
    # print(df[692:695])

    # Check empty
    # print((df.MESSAGE.str.len() == 0).any())

    # Locate empty
    # print(df(df.MESSAGE.str.len() == 0).index)
    # df.index.get_loc('.DS_Store')

    # Remove System File Entries from Dataframe
    # df = df.drop(['cmds', 'DS_Store'])
    # df.drop(['cmds', 'DS_Store'], inplace=True)

    # Add Document IDs to Track Emails in Dataset
    document_ids = range(0, len(df.index))
    df['DOC_ID'] = document_ids
    df['FILE_NAME'] = df.index
    df.set_index('DOC_ID', inplace=True)
    print(df.head())
    print(df.tail())

    return df


def save_data(df):
    df.to_json(DATA_JSON_FILE)


def get_data_from_json():
    df = pd.read_json(DATA_JSON_FILE)

    document_ids = range(0, len(df.index))
    df['DOC_ID'] = document_ids
    df['FILE_NAME'] = df.index
    df.set_index('DOC_ID', inplace=True)

    return df


def draw_pie_chart(df):
    print(df.CLASSIFICATION.value_counts())
    spam_count = df.CLASSIFICATION.value_counts()[1]
    ham_count = df.CLASSIFICATION.value_counts()[0]
    labels = ['Spam', 'Ham']
    sizes = [spam_count, ham_count]
    custom_colors = ['#c23616', '#487eb0']
    # offset = [0.05, 0.05]
    # labels = ['Spam', 'Ham', 'Lamb', 'Cam']
    # sizes = [30, 40, 20, 10]
    # custom_colors = ['#c23616', '#487eb0', '#e1b12c', '#4cd137']
    # offset = [0.05, 0.05, 0.05, 0.05]

    plt.figure(figsize=[3, 3], dpi=254)
    plt.pie(
        sizes,
        labels=labels,
        textprops={'fontsize': 9},
        startangle=90,
        autopct='%1.0f%%',
        colors=custom_colors,
        # explode=offset,
        pctdistance=0.8
    )

    # plt.show()

    # Donut Chart
    centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
    plt.gca().add_artist(centre_circle)

    plt.show()


def tokenize_message(message):
    return word_tokenize(message.lower())


def remove_stopwords(words_list):
    stopwords_set = set(stopwords.words('english'))
    return [stemmer.stem(word) for word in words_list if word.isalpha() and word not in stopwords_set]


def remove_html_tags(message):
    soup = BeautifulSoup(message, 'html.parser')
    return soup.prettify()
    # return soup.get_text()


msg = "All work and no play makes Jack a dull boy. To be or not to be. ??? Nobody expects the Spanish Inquisition!"
words = tokenize_message(msg)
filtered_words = remove_stopwords(words)
print(filtered_words)


def clean_data(df):
    pass


# data = get_data()
# save_data(data)
data = get_data_from_json()
print(data.head())
print(data.tail())

# draw_pie_chart(data)


# print(remove_html_tags(data.at[2, 'MESSAGE']))
