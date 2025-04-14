'''
Created on Aug 18, 2021

@author: xiaosonh
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image

import glob
  
from sklearn.model_selection import train_test_split
from labelme import utils

import asyncio
import time
import yaml
import re
import pathlib

def setClassIDZero(path):
    names = glob.glob(path+"/*.txt")
    
    for name in names:     
        content = ""   
        with open(name+"", 'r') as f:
            lines = f.readlines()
            for line in lines:
                row = line.split(" ")
                # print(len(row))
                content = content + "0 %s %s %s %s\n" % (row[1], row[2], row[3], row[4])
            f.close() 

        with open(name+"", 'w') as f:
            f.write(content)
            f.close()


def getItemIndex(path, shelfID, itemID):
    # ItemArray = []
    print(path + shelfID+ "/data.yaml")
    with open(path + shelfID+ "/data.yaml", 'r') as stream:
        # out = yaml.load(stream)
        # nc = (yaml.safe_load(stream)['nc'])
        # print('nc: '+str(nc))
        # data = yaml.load(stream)
        # ItemArray = data['names']

        symbolList = (yaml.safe_load(stream)['names'])
        # print(symbolList)
        # print('ItemID index at data.yaml: '+str(symbolList.index(itemID)))
        stream.close()  
        return symbolList.index(itemID)

def ArrangeExtraExpFiles(path, shelfID, itemID, objectIndex, type, targetDir):
    names = glob.glob(path+shelfID+"/labels/"+type+"/*.txt")
    ItemArray = []    

    for name in names:        
        with open(name+"", 'r') as f:
            lines = f.readlines()
            if len(lines) == 1 and lines[0].split()[0] == str(objIndex):
                name = name.replace("\\", "/")
                print(name)
                # print(name.rpartition('/')[-1])
                pureFileName = name.rpartition('/')[-1].replace('.txt', '')
                # print(lines[0]+"")
                figName = path+shelfID+"/images/"+type+"/%s.png" % pureFileName
                print('pureFileName: '+pureFileName)
                shutil.copyfile(name, targetDir+itemID+"/labels/"+type+'/'+pureFileName+'.txt')
                shutil.copyfile(figName, targetDir+itemID+"/images/"+type+'/'+pureFileName+'.png')

            f.close()                


def printf(format, *args):
    sys.stdout.write(format % args)

def dumpDataYAML(path, name):
    src = path + name + '/data.yaml'
    name = name.replace('YOLODataset', '')
#     os.mkdir('../dataYAML/' + name)
    dest = '../dataYAML/' + name + '/data.yaml'
    print(dest)
    shutil.copyfile(src, dest)    

def genPredictCode(path):
    print('python val.py --weights runs/train/%sYOLOv5s/weights/best.pt --img 416 --conf 0.4 --data ../MuseumImages/%s/data.yaml --verbose --save-conf --save-hybrid --name %s &&' 
          % (name, name, name))

def rmBlankFiles(path1, path2, path3):
    path = path1+path2+path3
    names = os.listdir(path)
    
    for name in names:    
        file_stats = os.stat(path+name)
        if file_stats.st_size == 0:
            labelName = name.replace('png', 'txt')
            print(f'{name} is zero file size')
            print(path1+'/labels'+path3+labelName)
            os.remove(path+name)
            os.remove(path1+'/labels'+path3+lableName)

def replaceBlankFileName(path):
    names = os.listdir(path)
    for name in names:    
        newName = name.replace(" ", "")
        
        if name != newName:
            os.rename(path+name, path+newName)        

def arrangeDataYAML(path, name):
    txtFile = path+name+"/data.yaml"
    lines = ""
    with open(txtFile, 'r') as f:
        lines = f.read()
        lines = lines.replace("train: ", "train: ../MuseumImages/")
        lines = lines.replace("val: ", "val: ../MuseumImages/")
        lines = lines.replace("\\", "/")            
        lines = lines.replace("YOLODataset", "")   
        lines = lines.replace(name+"/"+name, name)  
        # print(lines)
        f.close()    

    with open( txtFile, 'w') as f:
        f.write(lines)
        f.close()

def createModelYAML(path, name):
        # Try to read the number of classes
        print(path+"/"+name+"/"+name+"YOLODataset/data.yaml")
        num_classes1 = ''
        with open( path+"/"+name+"/"+name+"YOLODataset/data.yaml", 'r') as f:
            num_classes1 = str(yaml.safe_load(f)['nc'])        
            print(num_classes1)
            f.close()

        modelContent = ''    
        with open( "custom_yolov5s.yaml", 'rt') as f:
            modelContent = f.read()     
            modelContent = modelContent.replace('{num_classes}', num_classes1)  
            # modelContent = modelContent.format(num_classes = num_classes1)
            # print(modelContent)
            f.close()          
        
        outputFileName = name+"_yolov5s.yaml"
  
        with open( outputFileName, 'w') as f:
            f.write(modelContent)
            f.close()

if __name__ == '__main__':
    # path = os.getcwd()
    path = '../P2/'
    names = os.listdir(path)
    print(names)  

    # target = ["S203.C209.02.00"]
    # target = ["S203.C209.02.00", "S203.C209.03.00", "S203.C335.02.14", "S203.C336.02.04", "S203.C336.02.05", "S203.C336.03.00", "S203.C336.04.00"]

    for name in names:                
        if os.path.isdir(name):
            # Step 1. Generate label12yolo command. Run these command outside this Python program.
            # print("python labelme2yolo.py --json_dir "+name+" --val_size 0.2")

            # Step 2. Move to ArtfactsProcessed directory and create Model YAML
            # src = name + "/"+name+"YOLODataset"
            # dest = "../ArtfactsProcessed/"+name+"YOLODataset"
            # shutil.copytree(src, dest)
            # createModelYAML(path, name)

            # Step 3. reduce YOLODataset from folder name
            # src = "../ArtfactsProcessed/"+name+"YOLODataset"
            # dest = "../ArtfactsProcessed/"+name
            # os.rename(src, dest)            

            # Step 4. Replace blank file names
            # replaceBlankFileName('../ArtfactsProcessed/'+name+'/images/train/')
            # replaceBlankFileName('../ArtfactsProcessed/'+name+'/images/val/')
            # replaceBlankFileName('../ArtfactsProcessed/'+name+'/labels/train/')
            # replaceBlankFileName('../ArtfactsProcessed/'+name+'/labels/val/')            

            # Step 5. Fit the data directory on TWCC
            # src = "../ArtfactsProcessed/"
            # arrangeDataYAML(src, name)

            # Step 6. TWCC container train commands
            # printf('python train.py --img 416 --batch 96 --epochs 500 --data ../MuseumImages/%s/data.yaml --cfg ./models/%s_yolov5s.yaml --weights \'\' --name %sYOLOv5s --cache && \n', name, name,name)    

            # Step 7. Val command
            # printf('python val.py --weights runs/train/%sYOLOv5s/weights/best.pt --img 416 --conf 0.4 --data ../MuseumImages/%sYOLODataset/data.yaml --verbose --save-conf --save-txt --name %s && \n', name, name, name)            

            # Step 8. Dump data.yaml
            # dumpDataYAML(path, name)

            # Step 9. Remove YOLODataset in the folder name before upload dataset to TWCC
            # newName = name.replace("YOLODataset", "")
            # os.rename(path+name, path+newName)

            # Step 10. All in one model. We run mergeDataSet.py first.
            print("python labelme2yolo.py --json_dir "+name+" --val_size 0.2")



    # Step 11. Extra experiments
    # itemID = ["2002.005.0125","2000.001.0022","2001.001.0199","2002.004.0291","2004.028.1184","2004.028.1169","2005.008.1717","2004.028.1186","2003.008.0719","2003.008.0720","2003.008.0721","2003.008.0727","2012.019.0002","2002.004.0292","2003.001.0818","2001.001.0203","2003.001.0806","2012.023.0018","2001.001.0205","2001.001.0207","2003.001.0817","2002.008.0177","2003.001.0566","2001.001.0204","2002.004.0289","2002.004.0293","2003.001.0807","2001.001.0200","2004.028.1185","2003.001.0816","2003.001.0567","2003.008.0722","2003.008.0723","2003.008.0724","2003.008.0725","2003.008.0726","2003.008.0729","2003.008.0730","2003.008.0732","2003.020.0039"]
    # shelfID = ["S202.B224.02.00","S202.B333.01.10","S202.B333.01.10","S202.B333.01.10","S202.B333.01.10","S203.C334.02.04","S203.C334.02.04","S203.C334.02.07","S203.C334.02.08","S203.C334.02.08","S203.C334.02.08","S203.C334.02.08","S203.C334.02.11","S203.C334.02.13","S203.C334.02.13","S203.C334.02.14","S203.C334.02.14","S203.C334.03.03","S203.C335.02.03","S203.C335.02.03","S203.C335.02.04","S203.C335.02.07","S203.C335.02.07","S203.C335.02.08","S203.C335.02.08","S203.C335.02.08","S203.C335.02.08","S203.C335.02.10","S203.C335.02.10","S203.C335.02.12","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C336.02.02"]
    itemID = ["2000.001.0022","2001.001.0199","2001.001.0200","2001.001.0203","2001.001.0204","2001.001.0205","2001.001.0207","2002.004.0289","2002.004.0291","2002.004.0292","2002.004.0293","2000.001.0021","2002.008.0177","2003.001.0566","2003.001.0567","2003.001.0806","2003.001.0807","2003.001.0816","2003.001.0817","2003.001.0818","2003.008.0719","2003.008.0720","2003.008.0721","2003.008.0722","2003.008.0723","2003.008.0724","2003.008.0725","2003.008.0726","2003.008.0727","2003.008.0729","2003.008.0730","2003.008.0732","2003.020.0039","2004.028.1169","2004.028.1184","2004.028.1185","2004.028.1186","2005.008.1717","2012.019.0002","2012.023.0018"]
    shelfID = ["S203.C334.02.03","S203.C334.02.03","S203.C335.02.10","S203.C334.02.14","S203.C335.02.08","S203.C335.02.03","S203.C335.02.03","S203.C335.02.08","S203.C334.02.03","S203.C334.02.13","S203.C335.02.08","S203.C334.02.13","S203.C335.02.07","S203.C335.02.07","S203.C335.02.14","S203.C334.02.14","S203.C335.02.08","S203.C335.02.12","S203.C335.02.04","S203.C334.02.13","S203.C334.02.08","S203.C334.02.08","S203.C334.02.08","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C334.02.08","S203.C335.02.14","S203.C335.02.14","S203.C335.02.14","S203.C336.02.02","S203.C334.02.04","S203.C334.02.03","S203.C335.02.10","S203.C334.02.07","S203.C334.02.04","S203.C334.02.11","S203.C334.03.03"]
    
    for i in range(len(itemID)):
        # if itemID[i] == '2003.001.0567' or itemID[i] == '2003.001.0806':
        #     continue

        # objIndex = getItemIndex(path, shelfID[i], itemID[i])
        targetDir = '../fourtyItems/'

    #     # ArrangeExtraExpFiles        
        # pathlib.Path(targetDir+"/"+itemID[i]+"/").mkdir(parents=True, exist_ok=True) 
        # pathlib.Path(targetDir+"/"+itemID[i]+"/images/").mkdir(parents=True, exist_ok=True) 
        # pathlib.Path(targetDir+"/"+itemID[i]+"/labels/").mkdir(parents=True, exist_ok=True) 
        # pathlib.Path(targetDir+"/"+itemID[i]+"/images/train/").mkdir(parents=True, exist_ok=True) 
        # pathlib.Path(targetDir+"/"+itemID[i]+"/images/val/").mkdir(parents=True, exist_ok=True) 
        # pathlib.Path(targetDir+"/"+itemID[i]+"/labels/train/").mkdir(parents=True, exist_ok=True)     
        # pathlib.Path(targetDir+"/"+itemID[i]+"/labels/val/").mkdir(parents=True, exist_ok=True)   
        # ArrangeExtraExpFiles(path, shelfID[i], itemID[i], objIndex, 'train', targetDir)
        # ArrangeExtraExpFiles(path, shelfID[i], itemID[i], objIndex, 'val', targetDir)

        # data.yaml
        # modelContent = ''
        # with open( "dataTemplate.yaml", 'r') as f:
        #     modelContent = f.read()        
        #     modelContent = modelContent.replace('ItemID', itemID[i])  
        #     f.close()
        
        # with open( targetDir+itemID[i]+"/data.yaml", 'w') as f:
        #     f.write(modelContent)
        #     f.close()    

        # Let the label class index to be 0
        # setClassIDZero(targetDir+"/"+itemID[i]+"/labels/train/")
        # setClassIDZero(targetDir+"/"+itemID[i]+"/labels/val/")

        # modelContent = ''
        # with open( "custom_yolov5s.yaml", 'r') as f:
        #     modelContent = f.read()        
        #     modelContent = modelContent.replace('{num_classes}', '1')  
        #     f.close()
        
        # with open( 'Fourty'+itemID[i]+"yolov5s.yaml", 'w') as f:
        #     f.write(modelContent)
        #     f.close()            
        
        # replaceBlankFileName('../fourtyItems/'+itemID[i]+'/images/train/')
        # replaceBlankFileName('../fourtyItems/'+itemID[i]+'/images/val/')
        # replaceBlankFileName('../fourtyItems/'+itemID[i]+'/labels/train/')
        # replaceBlankFileName('../fourtyItems/'+itemID[i]+'/labels/val/')           

        # Generate train scripts
        # printf('python train.py --img 416 --batch 96 --epochs 300 --data ../fourtyItems/%s/data.yaml --cfg ./models/Fourty%s_yolov5s.yaml --weights \'\' --name Fourty%sYOLOv5s --cache && \n', itemID[i], itemID[i],itemID[i])
        # printf('python train.py --img 416 --batch 96 --epochs 300 --data ../fourtyItems/%s/data.yaml --cfg ./models/Fourty%s_yolov5s.yaml --name Fourty%sYOLOv5s --cache && \n', itemID[i], itemID[i],itemID[i])
        
    # Generate val scripts
    # for i in range(len(itemID)):
    #     for j in range(len(itemID)):
    #         if i < j:         
    #             # print(i, j)
    #             modelFromItem = itemID[i]       
    #             testItem = itemID[j]
                # printf('python val.py --weights runs/train/Fourty%sYOLOv5s/weights/best.pt --img 416 --conf 0.4 --data ../fourtyItems/%s/data.yaml --verbose --save-conf --save-txt --name fourtyItems-%s-%s && \n', modelFromItem, testItem, modelFromItem, testItem)            

    # Extract prediction results
    confidenceString = ""
    for i in range(len(itemID)):
        for j in range(len(itemID)):
            if i < j:         
                # print(i, j)
                path = 'D:\\FourtyItemsResults\\fourtyItems-'+itemID[i]+"-"+itemID[j]+"\\labels\\"
                names = glob.glob(path+"*.txt")
                confidence = 0
                counter = 0.0
                for name in names:
                    # print(name)
                    with open(name+"", 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            # print(line)
                            row = line.split(" ")
                            confidence = confidence + float(row[5])
                            counter = counter + 1.0
                            # print(str(counter) +" "+str(confidence))
                        f.close()     

                if counter == 0.0:
                    avgConfidence = 0
                else:
                    avgConfidence = confidence / counter    
                # print(itemID[i]+" "+itemID[j]+" "+str(avgConfidence)+"\n")      
                confidenceString = confidenceString + itemID[i]+" "+itemID[j]+" "+str(avgConfidence) +"\n"    

    with open("confidence.txt", 'w') as f:
        f.write(confidenceString)
        f.close()








        

        
        
