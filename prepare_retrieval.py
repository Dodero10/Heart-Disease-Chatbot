import shutil
import rank_bm25
import os
from rank_bm25 import BM25Okapi

def get_bm25_scores(question):
    # Get all file names and contents from the data folder
    file_names = []
    corpus = []
    for file_name in os.listdir('./data'):
        with open(f'./data/{file_name}', 'r', encoding='cp437') as f:
            doc = f.readlines()

        file_names.append(" ".join(file_name.split("-")))
        corpus.append(" ".join(doc))

    # Tokenize the titles
    titles = file_names
    words = [
        [word for word in doc.split()]
        for doc in titles
    ]

    # Calculate BM25 scores
    bm25 = BM25Okapi(words)
    indexs = list(range(len(words)))
    tokenized_query = question.split()
    a = bm25.get_top_n(tokenized_query, indexs, n=5)

    # Create or clear the retrieval folder
    folder_path = 'retrieval'
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder_path)

    # Save the top 5 files to the retrieval folder
    for i in range(len(a)):
        file_path = os.path.join(folder_path, titles[a[i]])
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(corpus[a[i]])
