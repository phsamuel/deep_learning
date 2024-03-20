import sys
from tokenize import group
import ast
# from torchsummary import summary
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot as pyQtSlot
# from PyQt5.QtCore import QTimer,QDateTime,Qt,QRect
# from PyQt5.QtGui import QPainter, QPen, QFont #, QWebPage # QWebEnginePage.HighlightAllOccurrences
# from PyQt5.QtWebEngineWidgets import QWebEnginePage

from PyQt5.QtWidgets import QFileDialog
from explore_networks import Ui_MainWindow
import os
import numpy as np
import bibtexparser
# import Qdesktopservice
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

from urllib.request import urlopen, Request
from json import loads as json_loads
from PyQt5.QtWebKitWidgets import QWebView,QWebPage
from datetime import datetime, timedelta
import torch
from torchvision import models
from os.path import exists
from json import dumps as json_dumps
import json

# -*- coding: utf-8 -*-

# Sample Python code for youtube.subscriptions.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

# so so freaking stupid, rather than google, should first read the error message first. this code support desktop 
# rather than web application. So when creating oauth credential, need to select  

import os

json_name = 'torch.json'

def query(url):
    req=Request(url)
    response=urlopen(req)
    return json_loads(response.read())

def reprocess_models():
    valid_models = {}
    for m in [m for m in dir(models) if m[0].islower()]:
        try:
            model=getattr(models,m)()
            st = model.eval()
            valid_models[m]=st
        except:
            print(f'cannot process {m}')

    with open(json_name,'w') as f:
        f.write(json_dumps(valid_models))





class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.comboBox.clear()
        self.model=None

        with open('torchvision-models.json') as f:
            self.evalD = json.load(f)
        for c in sorted([m for m in self.evalD]): #('net' in m.lower() or 'vgg' in m.lower()) and (m[0].islower())]):
            self.comboBox.addItem(c)




        # channel_list = YT_all_subscriptions(self.youtube)
        # # end get current subscriptiondayss

        # self.timethreshold=datetime.now()+timedelta(weeks=-100*52) #100 years ago, default, i.e., no filter

        # self.my_cat = ['FAVORITE','NEWS','MISC','MATH','SCIENCE']
        # self.title_dict = {c[0]:c[1] for c in channel_list}
        # self.sub_dict = {c[0]:c[2] for c in channel_list}
        # self.channel_list = [c[0] for c in channel_list]
        # self.channel_with_cat,unclassified_channels = group_channels_by_categories(self.categories,self.channel_list)
        # self.channel_with_cat['Unclassified']=unclassified_channels
        # with open('youtube.my_cat') as f:
        #     self.channel_with_cat.update(json_loads(f.read()))
        # # print(self.channel_with_cat)
        # # with open('youtube.fav') as f:
        # #     self.channel_with_cat['FAVORITE']=[ch for ch in f.read().split(',') if ch!='']
        # # with open('youtube.news') as f:
        # #     self.channel_with_cat['NEWS']=[ch for ch in f.read().split(',') if ch!='']
        # # self.channel_with_cat['MISC']=[] # dummy

        # self.statusbar.showMessage(f'Number of channels: {len(self.channel_list)}')


    @pyQtSlot()
    def on_reloadButton_pressed(self):
        self.reload_channels()
            

    @pyQtSlot(str)
    def on_comboBox_currentIndexChanged(self, t):
        # print(self.channel_with_cat)
        try:
            self.model=getattr(models,t)()
            st = self.model.eval()
            self.evalBrowser.setHtml(str(st).replace('\n','\n<br>\n'))
        except ValueError as e:
            print(e)

        try:
            self.input_size=tuple(ast.literal_eval(self.inputDimEdit.toPlainText()))
            # buffer= StringIO()
            st= summary(self.model,input_size=self.input_size,device='cpu')
            self.summaryBrowser.setHtml(str(st).split('Total params')[0].replace('\n','\n<br>\n'))
            self.paramBrowser.setHtml('Total param'+str(st).split('Total params')[-1].replace('\n','\n<br>\n'))
        except ValueError as e:
            print(e)

        try:
            writer = SummaryWriter('torchvision_model')

            # writer.add_graph(net, images)
            writer.add_graph(self.model,torch.zeros(self.input_size))
            writer.close()
        except ValueError as e:
            print(e)


    @pyQtSlot()
    def on_searchEdit_textChanged(self):
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        t = self.searchEdit.toPlainText()
        for c in sorted([m for m in self.evalD if t.lower() in self.evalD[m].lower()]): 
            self.comboBox.addItem(c)

        print("helloworld")
        self.comboBox.blockSignals(False)
            
    # @pyQtSlot(str)
    # def on_inputDimEdit_textChanged(self,text):
    #     print(text)
    #     self.input_size=tuple(ast.literal_eval(text))


    #     try:
    #         st=summary(self.model,input_size=self.input_size,device='cpu')
    #         self.summaryBrowser.setHtml(str(st).replace('\n','\n<br>\n'))
    #     except ValueError as e:
    #         print(e)

    @pyQtSlot()
    def on_pushButton_pressed(self):
        reprocess_models()

        print('hello')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())

