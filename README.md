# Aplicação de redes neurais profundas na caracterização de rejeitos radioativos

Neste repositório estão armazenados os codigos Fortran, Pytho, Bash e outros utilizados no meu trabalho de mestrado com a seguinte estrutura:

## Aplitavito

Aplitativo Python para utilizacao para classificacao de espectros gama salvos em formato IEC. O Aplicativo carrega a rede neural treinada e realiza a inferencia (classificacao e regressao).

## PenEASY

Código Fortran para o tally que simula o multicanal do detector. Essa versão recebe como parâmetro a curva de calibração a ser utilizada no detector.


## Simulações

Scripts python e bash para a criação dos arquivos de configuração do PenEASY, execução e monitoramento das simulações e consolidação dos dados de saída pelo PenEASY.

## Modelagem

Scripts python para a criacao de redes neurais (utilizando Keras), treino, avaliação de desempenho e treino final da rede neural que realiza classificação e regressão.