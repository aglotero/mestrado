import tkinter as tk
import tkinter.ttk as ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils import carrega_dados_iec, carrega_dados_penelope, model_predict
import tensorflow as tf
from keras.models import load_model

def accuracy_score_wrapper(label, pred):
    threshold = .95
    label = tf.dtypes.cast((label >= threshold), tf.uint8)
    pred = tf.dtypes.cast((pred >= threshold), tf.uint8)
    return np.float32(accuracy_score(label, pred))

def my_accuracy_score(label, pred):
    metric_value = tf.compat.v1.py_function(accuracy_score_wrapper, [label, pred], tf.float32)
    return metric_value

class AppApp:
    def __init__(self, master=None):
        # load model
        self.model = load_model('VGG-19-Adam-classificacao-regressao-PENELOPE-v4_atividade.hdf5',
        custom_objects={'my_accuracy_score': my_accuracy_score})
        # build ui
        frmMain = ttk.Frame(master)

        btnCarregarEspectro = ttk.Button(frmMain)
        btnCarregarEspectro.config(text='Carregar Espectro')
        btnCarregarEspectro.grid()
        btnCarregarEspectro.columnconfigure('0', minsize='0')
        btnCarregarEspectro.configure(command=self.btnCarregarEspectro_click)

        btnCarregarPenelope = ttk.Button(frmMain)
        btnCarregarPenelope.config(compound='left', text='Carregar Simulação penEasy')
        btnCarregarPenelope.grid(column='1', columnspan='4', row='0')
        btnCarregarPenelope.configure(command=self.btnCarregarPenelope_click)

        btnInferencia = ttk.Button(frmMain)
        btnInferencia.config(compound='left', text='Processar')
        btnInferencia.grid(column='5', row='0')
        btnInferencia.configure(command=self.btnInferencia_click)

        imgEspectro = tk.Canvas(frmMain, name="imgEspectro")
        #imgEspectro.config(background='#d936d9')
        imgEspectro.grid(column='0', columnspan='5', row='2', rowspan='1')
        imgEspectro.columnconfigure('0', minsize='0')

        self.figEspectro, self.axEspectro = plt.subplots(1, 2) #, figsize=(7, 4), dpi=100
        self.axEspectro[0].axis("off")
        self.axEspectro[1].axis("off")
        self.canvasEspectro = FigureCanvasTkAgg(self.figEspectro, master=imgEspectro)
        self.canvasEspectro.draw()
        self.canvasEspectro.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbarEspectro = NavigationToolbar2Tk(self.canvasEspectro, imgEspectro)
        toolbarEspectro.update()
        self.canvasEspectro.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        imgSaida = tk.Canvas(frmMain, name="imgSaida")
        #imgSaida.config(background='#d9d90c')
        imgSaida.grid(column='7', columnspan='5', row='2', rowspan='1')

        self.figSaida, self.axSaida = plt.subplots()
        self.axSaida.axis("off")
        self.canvasSaida = FigureCanvasTkAgg(self.figSaida, master=imgSaida)
        self.canvasSaida.draw()
        self.canvasSaida.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvasSaida, imgSaida)
        toolbar.update()
        self.canvasSaida.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        txtNomeArquivo = tk.Text(frmMain, name="txtNomeArquivo")
        txtNomeArquivo.config(height='1', width='50')
        _text_ = '''Selecione um arquivo para começar	'''
        txtNomeArquivo.insert('0.0', _text_)
        txtNomeArquivo.grid(column='0', columnspan='5', row='1')
        frmMain.config(height='800', width='1000')
        frmMain.grid(column='10', row='10')

        # Main widget
        self.mainwindow = frmMain

    def btnCarregarEspectro_click(self):
        try:
            filename = tk.filedialog.askopenfilename(filetypes = (("IEC files", ['*.IEC', '*.iec'])
                                                            ,("All files", "*.*") ))
            if not filename:
                return
            
            self.dados = carrega_dados_iec(filename)

            #fig, axarr = plt.subplots(1, 2, figsize=(7, 4), dpi=100)
            self.axEspectro[0].clear()
            self.axEspectro[1].axis("off")

            self.axEspectro[0].plot(self.dados.channel, self.dados.counts, 'r')
            self.axEspectro[0].set_yscale("log")

            self.axEspectro[1].imshow(self.dados.counts.values.reshape((128,128)))

            self.canvasEspectro.draw()
            self.canvasEspectro.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            tk.messagebox.showerror("Mensagem", "Algo de errado não está certo:\n{0}".format(e))

    def btnCarregarPenelope_click(self):
        pass

    def btnInferencia_click(self):
        try:
            df = model_predict(self.model, self.dados.counts.values)
            df.update(df[['nuclei_score', 'nuclei_activity']].applymap('{:,.2f}'.format))

            val1 = ['probability', 'activity', 'activity uncertainty'] 
            val2 = df.radionuclideo.values 
            val3 = list(zip(df.nuclei_score, df.nuclei_activity, df.uncertainty)) 
            
            #self.fig.clf()
            self.axSaida.clear()

            #self.fig, self.ax = plt.subplots()
            self.axSaida.set_axis_off() 
            table = self.axSaida.table( 
                cellText = val3,  
                rowLabels = val2,  
                colLabels = val1,
                rowColours =["palegreen"] * 10,  
                colColours =["palegreen"] * 10, 
                colWidths=[.33]*3,
                cellLoc ='center',  
                loc ='upper left'
            )
            #table.auto_set_font_size(False)
            table.set_fontsize(12)
           
            self.canvasSaida.draw()
            self.canvasSaida.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            tk.messagebox.showerror("Mensagem", "Algo de errado não está certo:\n{0}".format(e))

    def run(self):
        self.mainwindow.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    app = AppApp(root)
    app.run()

