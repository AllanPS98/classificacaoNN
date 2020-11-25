from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.datasets.classification import ClassificationDataSet
import csv

dataset = csv.reader(open("dataset/breast-cancer-treino.csv", "r"))
ds = ClassificationDataSet(9, 1)
for row in dataset:
    row = [int(val) for val in row]
    ds.addSample(row[1:10], row[10])
nn = buildNetwork(ds.indim, 7, ds.outdim, bias=True)
treinador = BackpropTrainer(nn, ds)

print("Construiu a rede")

for i in range(100):
    treinador.train()

print("Treinou")

def classifica_e_calcula(limiar):
    dataset_test = csv.reader(open("dataset/breast-cancer-teste.csv", "r"))
    gabarito = csv.reader(open("dataset/breast-cancer-gabarito.csv", "r"))
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    total = 0
    list_resp = []
    for row_teste in dataset_test:
        total += 1
        if nn.activate(row_teste[1:10])[0] > limiar:
            list_resp.append(4)
        else:
            list_resp.append(2)
    for gab, resp in zip(gabarito, list_resp):
        gab = [int(val) for val in gab]
        if resp == gab[1] and gab[1] == 2:
            tn += 1
        elif resp == gab[1] and gab[1] == 4:
            tp += 1
        elif resp != gab[1] and gab[1] == 2:
            fn += 1
        elif resp != gab[1] and gab[1] == 4:
            fp += 1
    # acurácia
    acc = (tn + tp) / total
    # taxa de verdadeiros positivos: é a porcentagem de casos positivos corretamente classificados como pertencentes
    # à classe positiva.
    tp_rate = tp / (tp + fn)
    # taxa de verdadeiros negativos: é a porcentagem de casos negativos corretamente classificados como pertencentes
    # à classe negativa.
    tn_rate = tn / (fp + tn)
    # taxa de falsos positivos: é a porcentagem de casos negativos incorretamente classificados como pertencentes
    # à classe positiva.
    fp_rate = fp / (fp + tn)
    # taxa de falsos negativos: é a porcentagem de casos positivos incorretamente classificados como pertencentes
    # à classe negativa.
    fn_rate = fn / (tp + fn)
    return acc, tp_rate, tn_rate, fp_rate, fn_rate


acc, tp_rate, tn_rate, fp_rate, fn_rate = classifica_e_calcula(3)

print("Acurácia =  {0:.2f}%  "
      "\nTaxa de verdadeiros positivos = {1:.2f}% "
      "\nTaxa de verdadeiros negativos = {2:.2f}% "
      "\nTaxa de falsos positivos = {3:.2f}% "
      "\nTaxa de falsos negativos =  {4:.2f}%"
      .format(acc * 100, tp_rate * 100, tn_rate * 100, fp_rate * 100, fn_rate * 100))
auc = (1 + tp_rate - fp_rate) / 2
print("Área sobre a curva ROC = {:.2f}%".format(auc * 100))
