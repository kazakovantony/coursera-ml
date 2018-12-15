import re
import numpy as np
from scipy.spatial import distance

f = open('../sentenceCosinusDistance/sentences.txt', 'r')


def split_low(line):
    return re.split('[^a-z]', line.lower())


not_empty_words = list(
    filter(lambda word: word,
           [word for sentence in f.readlines() for word in split_low(sentence)]))

print(not_empty_words)

# sentence1: I(0) am(1) Anton(2) and(3) I(0) work(4) at(5) EPAM(6) - 7 unique words
# sentence2: Oleski(8) works(9) with(10) me(11) at(5) EPAM(6) - 4 unique words


# идем по всем предложениям,
# берем первое, создаем словарь, создаем счетчик слов, идем по всем словам,
# берем первое, берем словарь и проверяем наличие слова,
# если оно есть переходим к следующему слову
# если его нет инкрементим счетчик, добавляем слово в словарь со счетчиком

f = open('../sentenceCosinusDistance/sentences.txt', 'r')
unique_word_occurrence = {}
word_unique_index = 0
for sentence in f.readlines():
    for word in split_low(sentence):
        if word and word not in unique_word_occurrence:
            unique_word_occurrence[word] = word_unique_index
            word_unique_index += 1

print(word_unique_index)

# can be refactor to functional style

# матрица n*d (2*11)
# n - число предложений (2),
# d - число уникальных слов(0 - 11)
# матрица вхождений j слово в i - предложение
# 2,1,1,1, ... all unique words occurrence
# 0,0,0,0

# создаем матрицу (колво предложений, на кол-во слов в словаре)
# берем уникальное слово из словаря
# идем по всем предложениям
# берем первое, создаем элемент (i, j) = 0, идем по всем словам
# берем первое, проверяем на равенство с уникальным
# если равно увеличиваем элемент (i = 1, j = 0) на 1
# в конце цикла по словам присваиваем элемент (i, j) в матрицу

f = open('../sentenceCosinusDistance/sentences.txt', 'r')
sentences = f.readlines()
n = sentences.__len__()
d = word_unique_index

unique_word_occurrences_matrix = np.zeros(shape=(n, d), dtype=int)
current_sentence = 0
for sentence in sentences:
    for word in split_low(sentence):
        if word:
            unique_word_occurrences_matrix[current_sentence, unique_word_occurrence[word]] += 1
    current_sentence += 1


dists = list()
first_row = None
current_row_index = 0
for current_row in unique_word_occurrences_matrix:
    if first_row is None:
        first_row = current_row
    else:
        dists.append((current_row_index, distance.cosine(first_row, current_row)))
    current_row_index += 1

dists.sort(key=lambda tup: tup[1])
print(dists)
