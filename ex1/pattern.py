import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import copy

class Checker:
    def __init__(self, resolution, tile_size):
        self.r = int(resolution)
        self.t = int(tile_size)
        self.output1 = np.ones((self.t*2,self.t*2))
        self.output = np.zeros((self.r,self.r))

    def draw(self):
        #self.output[1::2,::2] = 1
        #self.output[::2,1::2] = 1
        self.output1[:self.t:,:self.t:] = 0
        self.output1[self.t:2*self.t:,self.t:2*self.t:] = 0
        self.output = np.tile(self.output1,(int(self.r/(self.t*2)),int(self.r/(self.t*2))))

        return copy.copy(self.output)

    def show(self):
        plt.imshow(self.draw(),cmap='gray')
        plt.axis('off')
        plt.show()
        #print(self.draw())

class Circle:
    def __init__(self, resolution, radius, position):
        self.res = int(resolution)
        self.rad = int(radius)
        self.pos = tuple(position)
        self.output = np.zeros((self.res,self.res))

    def draw(self):
        #x = np.arange(self.pos[0]-self.rad,self.pos[0]+self.rad)
        #y = self.pos[1] + int(np.sqrt(self.rad**2 - (x - self.pos[0])**2))
        #self.output[x[0]:x[-1]:,-y[-1]:y[1]:] = 1
        #theta = np.arange(0, 2*np.pi, 0.01)
        #x = self.pos[0] + self.radius * np.cos(theta)
        #y = self.pos[1] + self.radius * np.sin(theta)
        x = np.arange(self.pos[0]-self.rad,self.pos[0]+self.rad+1)
        y = np.arange(self.pos[1]-self.rad,self.pos[1]+self.rad+1)
        X,Y = np.meshgrid(x,y)
        dist = np.sqrt((X - self.pos[0]*np.ones(x.shape))**2 + (Y - self.pos[1]*np.ones(y.shape))**2)
        #print(X,Y)
        #dist = ma.masked_where(dist >= self.rad, dist)
        dist[dist <= self.rad] = 1
        dist[dist > self.rad] = 0
        self.output[self.pos[1]-self.rad:self.pos[1]+self.rad+1:,self.pos[0]-self.rad:self.pos[0]+self.rad+1:] = dist
        return copy.copy(self.output)

    def show(self):
        plt.imshow(self.draw(),cmap='gray')
        plt.axis('off')
        plt.show()

