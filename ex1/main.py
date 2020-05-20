import numpy as np
import matplotlib.pyplot as plt
import pattern
import json
import scipy.misc
from PIL import Image
import os.path
import generator


def main():
    #obj = pattern.Checker(100, 25)
    #obj.show()
    # obj1 = pattern.Circle(1024,200,(512,256))
    # obj1.show()
    #
    # obj2 = np.load('/anaconda/envs/DLEx/ex1/reference_arrays/circle.npy')
    # obj3 = obj2 - obj1.draw()
    # b = np.nonzero(obj3)
    # print (b)
    # print(obj3)
    # x = np.load('/anaconda/envs/DLEx/ex1/exercise_data/0.npy')
    # img = Image.fromarray(x)
    # img = img.resize((25,25))
    # x = np.array(img)
    # print(x.shape)
    # with open('/anaconda/envs/DLEx/ex1/Labels.json','r') as load_f:
    # label_dict = json.load(load_f)
    # print(label_dict[str(11)])
    #
    # a = np.empty((32, 32, 3))
    # b = np.empty((32, 32, 3))
    # b = np.concatenate ((b,a))
    # print(b.shape)
    #
    # path = '/anaconda/envs/DLEx/ex1/exercise_data'
    # a = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
    # print(a)
    obj2 = generator.ImageGenerator('/anaconda/envs/DLEx/ex1/exercise_data/', '/anaconda/envs/DLEx/ex1/Labels.json', 12 ,[32,32,3])
    # for i in range(18) :
    #     obj.next(j)
    #     j=j+1
    print(obj2.next()[0])
    obj2.show()


if __name__ == '__main__':
    main()
