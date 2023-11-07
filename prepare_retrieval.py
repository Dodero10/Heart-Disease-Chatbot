import rank_bm25
import os

from rank_bm25 import BM25Okapi


def get_bm25_scores(question):

    # file_names = []
    # corpus = []
    # for file_name in os.listdir('./data'):
    #     with open(f'./data/{file_name}', 'r') as f:
    #         doc = f.readlines()
    #
    #     file_names.append(" ".join(file_name.split("-")))
    #     corpus.append(" ".join(doc))

    # write code for reading the files and storing them in the corpus variable
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
    # for file_name, doc in zip(file_names, corpus):
    #     lines = doc.split("\n")
    #     if len(lines) > 5:
    #         raw = lines[5]
    #         title = lines[5].split(":")[0].split("?")[0]
    #         titles.append(title)
    #     else:
    #         print(f'Document {file_name} does not have enough lines.')

    words = [
        [word for word in doc.split()]
        for doc in titles
    ]
    bm25 = BM25Okapi(words)
    indexs = list(range(len(words)))

    tokenized_query = question.split()

    a = bm25.get_top_n(tokenized_query, indexs, n=5)
    for i in range(len(a)):
        # Tạo đường dẫn đến folder 'retrieval'
        folder_path = 'retrieval'

        # Kiểm tra xem folder đã tồn tại hay chưa, nếu chưa thì tạo mới
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Tạo đường dẫn file bao gồm cả folder
        file_path = os.path.join(folder_path, titles[a[i]])

        # Mở file để ghi nội dung
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(corpus[a[i]])