import csv
import random

dataset = csv.reader(open("dataset/breast-cancer.csv", "r"))

maligno = 0
benigno = 0
total = 0
lista_dataset = []
for row in dataset:
    if int(row[10]) == 2:
        benigno += 1
    else:
        maligno += 1
    lista_dataset.append(row)
    total += 1

print("maligno", maligno / total)
print("benigno", benigno / total)


for i in lista_dataset:
    if '?' in i:
        i[6] = str(random.randint(1, 10))


escrita_treino = csv.writer(open('dataset/breast-cancer-treino.csv', 'w', newline=''))
escrita_teste = csv.writer(open('dataset/breast-cancer-teste.csv', 'w', newline=''))
gabarito = csv.writer(open('dataset/breast-cancer-gabarito.csv', 'w', newline=''))
for x in range(0, 488):
    escrita_treino.writerow(lista_dataset[x])
for y in range(488, 699):
    escrita_teste.writerow(lista_dataset[y][:10])
    gabarito.writerow([lista_dataset[y][0], lista_dataset[y][10]])
