import os.path
import json
# import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False,
                 shuffle=True):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        # global batch_size, image_size, rotation, mirroring, shuffle, file_num, batch_num
        self.file_path = str(file_path)
        self.label_path = str(label_path)
        self.batch_size = int(batch_size)
        self.image_size = list(image_size)
        self.rotation = bool(rotation)
        self.mirroring = bool(mirroring)
        self.shuffle = bool(shuffle)
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_num = len([lists for lists in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, lists))])

        self.batch_num = int(self.file_num / self.batch_size)
        if self.file_num % self.batch_size != 0:
            self.batch_num = self.batch_num + 1

        self.index = list(range(self.file_num))
        self.c = 0

        with open(self.label_path, 'r') as load_f:
            self.label_dict = json.load(load_f)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        global batch, labels
        batch = np.empty((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        labels = np.empty(self.batch_size)

        if self.shuffle:
            if self.c == self.batch_num or self.c == 0:
                np.random.shuffle(self.index)
                self.c = 0
            else:
                pass

        for i in range(self.batch_size):
            img = np.load(self.file_path + str(self.index[0]) + '.npy')

            labels[i] = self.label_dict[str(self.index[0])]

            self.index.insert(self.file_num, self.index[0])
            self.index.remove(self.index[0])

            img = Image.fromarray(img)
            img = img.resize((self.image_size[0], self.image_size[1]))

            if self.mirroring:
                a = np.random.randint(0, 2, 1)
                if a == 1:
                    img = ImageOps.mirror(img)
            if self.rotation:
                a = np.random.randint(0, 2, 1)
                if a == 1:
                    img = img.rotate(180)

            batch[i] = np.array(img)/255

        self.c = self.c + 1

        return batch, labels

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        for i in range(self.batch_size):
            plt.subplot(5, 5, i + 1)
            plt.imshow(batch[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
        plt.show()


