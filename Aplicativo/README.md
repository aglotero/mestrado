# Aplicativo

O aplicativo carrega em memória a rede neural e permite que um usuário realize a inferência utilizando um arquivo de espectrometria gama armazenado no formato IEC.

## Execução

Arnazebar, no mesmo diretório do arquivo python, o arquivo binário contendo a rede neural treinada em formato hdf5 do Keras.

Executar no terminal:

```
python app.py
```

Em alguns segundos a interface gráfica será exibida.

## Saída

O aplicativo permite visualizar as saídas da rede neural em forma de tabela, onde é informada a probabilidade de ocorreência e atividade estimada de cada radionuclídeo.
