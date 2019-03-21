#!/usr/bin/env python
# coding=utf-8
"""
author: huangchang
file: mian.py
data: 2017.7.12
note: scene recognition gui tool
"""
from __future__ import print_function
import argparse
import os
import sys
if 2 == sys.version_info[0]:
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import urllib
else:
    import urllib.parse as urllib
    
import json
import cv2

if '2' == cv2.__version__[0]:
    cv_version = 2
else:
    cv_version = 3
    
import numpy as np
from reportlab.pdfgen import canvas

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor

class mainwindow(QMainWindow):
    """main window
    """
    value_changed = pyqtSignal(object)
    value_warning = pyqtSignal(str)

    def __init__(self, args):
        super(mainwindow, self).__init__()
        self.setAcceptDrops(True)

        self.createLayout()
        self.createActions()
        self.createMenus()

        self.setWindowTitle("行为识别 demo")
        self.setGeometry(100, 100, 1000, 800)

        self.args = args
        self.result_list = []
        self.scene_number = []
        self.scene_name = []
        self.index = 0

        # init model
        self.value_changed.connect(self.update_img)
             
    def createtext(self):
        create_text_dir = QFileDialog.getExistingDirectory(self,"批量测试目录")
        
        j = 0
        sub_text_path = []
        
        for directory, folder, files in os.walk(create_text_dir):
            if j > 0:
                for file in files:
                    file_list = file.split('.')
                    file_format = file_list[-1]
                    
                    if file_format == 'txt':
                        sub_text_path.append(os.path.join(directory,file))
            j += 1
        
        self.result_list = []
        
        for i in range(len(sub_text_path)):
            
            self.result_list.extend(self.load_json(sub_text_path[i]))
        
            
        temp_result = []
        scene_set = set()
        
        for i in range(len(self.result_list)):
            scene_set.add(self.result_list[i][3])
        
        temp_scene_list = []
        temp_scene_list.extend(scene_set)
        
        scene_list = sorted(temp_scene_list)
        
        for i in scene_list:
            for j in range(len(self.result_list)):
                if i == self.result_list[j][3]:
                    temp_result.append(self.result_list[j])
                    
        for i in range(len(temp_result)):
            temp_result[i][0] = i
         
        self.result_list = []
        
        for i in temp_result:
            self.result_list.append(i) 
            
        text_name = 'batch_dir_path.txt'
        
        text_dir = os.path.join(create_text_dir,text_name)
        
        with open(text_dir,'w') as f:
            
            for i in range(len(self.result_list)):
                
                if i == len(self.result_list)-1:
                    f.write('{},{},{},{}'.format(self.result_list[i][0],self.result_list[i][3],
                            self.result_list[i][4],self.result_list[i][5]))
                else:
                    f.write('{},{},{},{}\n'.format(self.result_list[i][0],self.result_list[i][3],
                            self.result_list[i][4],self.result_list[i][5]))
                
    def load_json(self,text_path):
        with open(text_path,'r') as f:
            json_str = json.load(f)
        if json_str is None or not json_str:
            return
        items = sorted(json_str.items())
        result_total_list = []
        result_list = []
        for key, obj in items:
            obj_list = []
            sub_items = sorted(obj.items())
            for sub_key, sub_obj in sub_items:
                obj_list.append([int(sub_obj[0]), int(sub_obj[1]), float(sub_obj[2]), sub_obj[3], float(sub_obj[4]), sub_obj[5]])
            result_total_list.append(obj_list)
        
        result_two_list = []
        for i in range(len(result_total_list)):
            if len(result_total_list[i]) > 1:
                result_two_list.append(result_total_list[i])
                
        for i in result_two_list:
            if i[0][0] == 1:
                result_list.append(i[0])
                
        return result_list        
    
    def lodetext(self):
        self.batch_file,_ = QFileDialog.getOpenFileName(self,"载入批量识别结果文件",filter = "文本文件(*.txt)")
        if self.batch_file is None or not self.batch_file:
            return
        
        self.result_list_widget.clear()
        self.result_list = []
        
        f = open(self.batch_file) 
            
        while 1:
            line = f.readline()
            if not line or line == '':
                break

            l = line.strip('\n').split(',')
            ll = [l[0],l[1],l[2],l[3]]
            self.result_list.append(ll)
        
        self.update_list()
        
        temp_name = []
        temp_name.append(self.result_list[0][1])
        
        for i in range(len(self.result_list)):
            if i < len(self.result_list)-1:
                
                if self.result_list[i][1] == self.result_list[i+1][1]:
                    pass
                else:
                    temp_name.append(self.result_list[i+1][1])
        
        self.scene_name.extend(temp_name)
        
    def write_result(self):
        
        number = 0  
        temp_number = []
         
        for i in self.scene_name:
            for j in range(len(self.result_list)):
                if i == self.result_list[j][1]:
                    number += 1
                    
            temp_number.append(number)
            number = 0
            
        with open(self.args.label_path,'r') as f:
            text_list = f.readlines()
            for index, item in enumerate(text_list):
                text_list[index] = item.strip('\n').split(':')[-1]
        
        total_result = []
        signal = 0
        
        for i in text_list:
            for index, item in enumerate(self.scene_name):
                if i == item:
                    total_result.append([i,temp_number[index],self.scene_number[index]])
                    signal = 1
            if signal == 1:
                pass
            else:
                total_result.append([i,0,0])
            signal = 0
                    
        accuracy_path = '{}.txt'.format(self.args.save_path)  

        with open(accuracy_path, 'w') as f:
            for i in range(len(total_result)):
                if i == len(total_result) - 1:
                    f.write('{}\t{}\t{}'.format(total_result[i][0],total_result[i][1],total_result[i][2]))
                else:
                    f.write('{}\t{}\t{}\n'.format(total_result[i][0],total_result[i][1],total_result[i][2]))
                    
    def pdf(self):
        
        temp_result_list = []
        a, _ = divmod(len(self.result_list),6)
        temp_result_list.append(self.result_list[0:a])
        temp_result_list.append(self.result_list[a:2*a])
        temp_result_list.append(self.result_list[2*a:3*a])
        temp_result_list.append(self.result_list[3*a:4*a])
        temp_result_list.append(self.result_list[4*a:5*a])
        temp_result_list.append(self.result_list[5*a:])
        
        k = 0
        
        for j in temp_result_list:
            
            c = canvas.Canvas('v2_199_8_3_{}.pdf'.format(k))
            i = 0
            long_limit = len(j)-5
            times, _ = divmod(len(j),6)
            for jj in range(times):
                
                if i <= long_limit:
                    image_path_0 = self.result_list[i][3]
                    label_str_0 = '{}:{:.3f}'.format(j[i][1],float(j[i][2]))
                    image_path_1 = j[i+1][3]
                    label_str_1 = '{}:{:.3f}'.format(j[i+1][1],float(j[i+1][2]))
                    image_path_2 = j[i+2][3]
                    label_str_2 = '{}:{:.3f}'.format(j[i+2][1],float(j[i+2][2]))
                    image_path_3 = j[i+3][3]
                    label_str_3 = '{}:{:.3f}'.format(j[i+3][1],float(j[i+3][2]))
                    image_path_4 = j[i+4][3]
                    label_str_4 = '{}:{:.3f}'.format(j[i+4][1],float(j[i+4][2]))
                    image_path_5 = j[i+5][3]
                    label_str_5 = '{}:{:.3f}'.format(j[i+5][1],float(j[i+5][2]))
                    
                    c.drawImage(image_path_0,0,600,300,200)
                    c.drawString(0,585,label_str_0)
                    
                    c.drawImage(image_path_1,300,600,300,200)
                    c.drawString(300,585,label_str_1)
                    
                    c.drawImage(image_path_2,0,320,300,200)
                    c.drawString(0,305,label_str_2)
                    
                    c.drawImage(image_path_3,300,320,300,200)
                    c.drawString(300,305,label_str_3)
                    
                    c.drawImage(image_path_4,0,45,300,200)
                    c.drawString(0,30,label_str_4)
                    
                    c.drawImage(image_path_5,300,45,300,200)
                    c.drawString(300,30,label_str_5)
                    
                    c.showPage()
                    
                    i += 6
            
            c.save()
            k += 1
    
    def createActions(self):
        """create actions """
        self.batchcreate = QAction("&生成",self,statusTip="Create the batch text",triggered=self.createtext)
        self.batchload = QAction("&载入",self,statusTip="Load the batch text",triggered=self.lodetext)
        
        self.createpdf = QAction("&生成pdf",self,statusTip="Create the pdf", triggered = self.pdf)
        
    def createMenus(self):
        self.batchMenu = self.menuBar().addMenu("&批量统计")
        self.batchMenu.addAction(self.batchcreate)
        self.batchMenu.addAction(self.batchload)
        
        self.pdfMenu = self.menuBar().addMenu("&生成文档")
        self.pdfMenu.addAction(self.createpdf)
        
    def createLayout(self):
        self.layout1 = QHBoxLayout()
        self.image_label = QLabel()
        self.result_list_widget = QListWidget()

        self.layout1.addWidget(self.image_label)
        self.layout1.addWidget(self.result_list_widget)
        self.result_list_widget.itemSelectionChanged.connect(self.change_select_img)
        # self.result_list_widget.itemClicked.connect(self.show_select_img)
        self.result_list_widget.setFixedWidth(250)
        
        self.layout2 = QHBoxLayout()
        self.number = QLineEdit()
        self.number.returnPressed.connect(self.number_changed)
        self.click_button = QPushButton()
        self.click_button.setText('click')
        self.click_button.clicked.connect(self.click_changed)
        
        self.layout2.addWidget(self.number)    
        self.layout2.addWidget(self.click_button)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(self.layout1)
        mainLayout.addLayout(self.layout2)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(mainLayout)
    
    def click_changed(self):
        show_str = self.scene_name[self.index]
        self.number.setText('{}:'.format(show_str))
        self.index += 1
        
    def number_changed(self):
        
        if self.index < len(self.scene_name):
            
            
            show_str = self.scene_name[self.index]
            number = self.number.text().split(':')[-1]  
            try:
                one = float(number)
            except:
                number = 0

            self.scene_number.append(int(number))   
            self.number.setText('{}:'.format(show_str))
            self.index += 1
        
        else:
            number = self.number.text().split(':')[-1]
            try:
                one = float(number)
            except:
                number = 0
            
            self.scene_number.append(int(number))
         
            self.write_result()
            self.number.setText('all write ok !!!')
            self.index = 0
            self.scene_number = []
    
    def change_select_img(self):
        item = self.result_list_widget.currentItem()
        self.show_select_img(item)

    def show_select_img(self, item):
        # bool_valid, frame_id, self.cur_time, result[0], result[1], img.copy()
        text = item.text().strip().split(',')
        if len(text) < 1:
            return
        sub_index = text[1].split('_')
        cur_result = self.result_list[int(sub_index[1])]
        sub_text = text[0].split(':')
        text = "{},{:.3f}" \
            .format(sub_text[0], float(sub_text[1]))
        cur_img_file = cur_result[3]
        img = cv2.imread(cur_img_file)
        cv2.putText(img, text, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),2)
        self.update_img(img)

    def update_img(self, img):
        if img is not None and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, depth = img_rgb.shape
            qimg = QImage(img_rgb, w, h, QImage.Format_RGB888)
            pimg = QPixmap.fromImage(qimg)
            pimg2 = pimg.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pimg2)

    def update_list(self):
        self.result_list_widget.setFixedWidth(250)
        self.result_list_widget.clear()
        print(len(self.result_list))
        for result in self.result_list:
            item = QListWidgetItem('{}:{},{}_{}'.format(result[1],result[2],1,result[0]))
                    
            self.result_list_widget.addItem(item)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="行为识别 demo")

    parser.add_argument('--save_path', type=str,
                        help='结果保存路径', default='results')
    parser.add_argument('--label_path',type=str,help='the label text path',default='model/labels_303_3_21.txt')
    return parser.parse_args(argv)


def main(args):
    app = QApplication(sys.argv)
    ex = mainwindow(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
