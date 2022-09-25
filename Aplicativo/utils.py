import numpy as np
import pandas as pd

nomes_nuclideos = {
            'Am-241': 0,
            'Ba-133': 1,
            'Cd-109': 2,
            'Co-57': 3,
            'Co-60': 4,
            'Cs-137': 5,
            'Eu-152': 6,
            'Mn-54': 7,
            'Na-22': 8,
            'Pb-210': 9
    }

def model_predict(model, data):
    preds = model.predict(np.array([data.reshape((128, 128, 1))]))

    df = pd.DataFrame({'radionuclideo': list(nomes_nuclideos.keys()),
                   'nuclei_score': np.round(preds[0][0]*100, 2),
                   'nuclei_activity': np.round(np.exp(preds[1][0]) - 1, 2),
                   'uncertainty': np.sqrt(np.sum(data))
                  })
    return df

def carrega_dados_iec(filename):

    with open(filename, 'r') as f:
        linhas = f.readlines()

    tempo = 1#float(linhas[1].split('   ')[1])

    dados = []
    pode_comecar = False
    for linha in linhas:
        if linha == "A004USERDEFINED                                                     \n":
            pode_comecar = True
            continue
        if pode_comecar:
            aux = linha.strip().split()
            dados.extend([float(x) / tempo for x in aux[2:]])

    return pd.DataFrame({'channel': range(0, len(dados[1:])), 'counts' : dados[1:]})


def carrega_dados_penelope(filename, n_particulas=1.0e07):
    data = {
        'Elow(eV)' : [],
        'Emiddle(eV)' : [],
        'counts(1/eV/hist)' : [],
        '+-2sigma':[],
        'nbin' : []
    }

    linha_de_dados = False
    with open(filename, "r") as f:
        for line in f:
            if line[0] == '#':
                linha_de_dados = False
            else:
                linha_de_dados = True

            if linha_de_dados:
                aux = line.split(' ')
                if len(aux) == 2:
                    break

                data['Elow(eV)'].append(np.fromstring(aux[2], dtype=np.float64, sep=',')[0])
                data['Emiddle(eV)'].append(np.fromstring(aux[4], dtype=np.float64, sep=',')[0])
                q = np.fromstring(aux[6], dtype=np.float64, sep=',')[0]
                data['counts(1/eV/hist)'].append(q)
                data['+-2sigma'].append(np.fromstring(aux[8], dtype=np.float64, sep=',')[0])
                nbin = np.fromstring(aux[9], dtype=np.int, sep=',')[0]
                data['nbin'].append(nbin)

    df = pd.DataFrame.from_dict(data)
    df['counts'] = df['counts(1/eV/hist)'].values * n_particulas * 1 / ((df['Elow(eV)'].shift(-1).fillna(0) - df['Elow(eV)']) / df['nbin'])
    df['counts'] = df['counts']#.astype(np.uint8)
    df['counts'][0:20] = 0
    df['E'] = df['Elow(eV)'] / 1e3
    return df.loc[0:16383]
