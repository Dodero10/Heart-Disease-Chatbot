import shutil

import rank_bm25
import os

from rank_bm25 import BM25Okapi


def get_bm25_scores(question):

    file_names = []
    corpus = []
    for file_name in os.listdir('./data'):
        with open(f'./data/{file_name}', 'r',encoding='cp437') as f:
            doc = f.readlines()

        file_names.append(" ".join(file_name.split("-")))
        corpus.append(" ".join(doc))

    # import matplotlib.pyplot as plt
    # plt.hist([len(doc.split()) for doc in corpus], bins=128, range=(0, 5000))
    # plt.show()

    titles = file_names

    words = [
        [word for word in doc.split()]
        for doc in titles
    ]
    bm25 = BM25Okapi(words)
    indexs = list(range(len(words)))

    tokenized_query = question.split()

    a = bm25.get_top_n(tokenized_query, indexs, n=5)
    folder_path = 'retrieval'

    # Kiểm tra nếu thư mục tồn tại
    if os.path.exists(folder_path):
        # Xóa tất cả các file trong thư mục
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
        # Nếu thư mục không tồn tại, tạo thư mục mới
        os.makedirs(folder_path)

    # Lưu file mới
    for i in range(len(a)):
        file_path = os.path.join(folder_path, titles[a[i]])

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(corpus[a[i]])