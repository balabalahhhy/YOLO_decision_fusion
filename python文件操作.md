### python文件操作	

#### 1.读取YOLO格式数据集标签演示

```python
file1 = open("D:/project_file/python_yolo/yolov5-5.0/runs/detect/exp28/labels/00001v.txt", "r")
lines1 = file1.readlines()#txt文件的每一行字符串都转化为列表的元素
# print(lines1)
for line in lines1:
    line.strip('\n')
    line = line.split()# 将字符串分割为列表默认分隔符为空格
    print(line)
    line2 = []
    for my_num in line:
        my_num1 = float(my_num)# 此种方法不能直接将字符转化为数字，需要经过中间文件
        line2.append(my_num1)
    print(line2)
```

将字符串列表转化为数字列表的方法

```python
    注意：
    # for my_num in line:
    #     my_num = float(my_num)# 此种方法不能直接将字符转化为数字,应该使用以下方法
    # 更加简单的方法
    line = [ float(x) for x in line ]#把字符串列表转化成数字列表
```

将YOLO格式转化为cv2格式，因为YOLO格式为x，y，w，h，需要将其转化为左上角坐标和右下角坐标

```python
def yolo_to_voc(size, box):
    x = box[1] * size[0]
    print(x)
    y = box[2] * size[1]
    w = box[3] * size[0]
    h = box[4] * size[1]
    xmin = int(x - w/2)
    xmax = int(x + w/2)
    ymin = int(y - h/2)
    ymax = int(y + h/2)
    return (xmin, ymin, xmax, ymax)
```

画出检测框，同时标注类和置信度

```python
def plot_labels(x, size, img):
    xmin, ymin, xmax, ymax = yolo_to_voc(size, x)
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax) ,(225,0,0))  # filled
    cv2.putText(img, class_img[int(x[0])] + str(x[5]) ,(xmin,ymin), 0, 0.7, (0,255,0), 1) 
```

### 完整代码

```python
from re import X
from matplotlib import lines
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def yolo_to_voc(size, box):
    x = box[1] * size[0]
    print(x)
    y = box[2] * size[1]
    w = box[3] * size[0]
    h = box[4] * size[1]
    xmin = int(x - w/2)
    xmax = int(x + w/2)
    ymin = int(y - h/2)
    ymax = int(y + h/2)
    return (xmin, ymin, xmax, ymax)

def plot_labels(x, size, img):
    xmin, ymin, xmax, ymax = yolo_to_voc(size, x)
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax) ,(225,0,0))  # filled
    cv2.putText(img, class_img[int(x[0])] + str(x[5]) ,(xmin,ymin), 0, 0.7, (0,255,0), 1) 

# 图片像素为630，460 YOLO格式坐标为0.150794 0.383696 0.225397 0.145652 为x,y,w,h
# 按行读取txt文件
size = [630, 460]
class_img = ['person', 'car']
x = [1, 0.150794,0.383696,0.225397,0.145652,0.437256] # x, y, w, h
img = cv2.imread('D:/project_file/test_iou_decision_fusion/test/images/00001v.jpg')
with open("D:/project_file/python_yolo/yolov5-5.0/runs/detect/exp28/labels/00001v.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        print(line)
        line = line.split()# 可以带空格的字符串，按空格给划分为列表
        labels = [ float(x) for x in line ]
        print(labels)
        plot_labels(labels, size, img)
cv2.imshow('my_img', img)
cv2.waitKey(0)
```

<img src="C:\Users\jiawenjie\Desktop\屏幕截图 2022-04-13 224227.png" alt="结果图片" style="zoom:50%;" />

