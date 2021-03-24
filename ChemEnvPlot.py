import plotly.graph_objects as go
import plotly.offline as offline
from plotly.offline import plot

from numpy import *
from mendeleev import element
from random import randint

import numpy as np
import torch

class ChemEnvironment:
    def __init__(self, 
                 xyz, 
                 Z=None, 
                 k=30, 
                 size=0.2, 
                 root=3
                ):
        # Todo: Add in spherical harmonics
        
        self.atomtypes = {0: 'None', 1: 'Hydrogen', 2: 'Carbon', 3: 'Nitrogen', 4: 'Oxygen', 5: 'Fluor'}
        self.k = k
        self.geometry = xyz.numpy()
        self.fig = None
        self.data = []
        
        # initialize basic sphere
        x_, y_, z_ = self.get_sphere() # default sphere radius = 1
        
        # set colors -> non-opaque greyscale for atom cores
        #            -> opaque colors for spherical harmonics
        self.colorscale = []
        element_list = self.get_elements(Z)
        
        if Z==None:
            # not functional atm
            r = size
            color = np.ones(self.geometry.shape[0])
            x_r, y_r, z_r = x_*r, y_*r, z_*r
            for i in range(self.geometry.shape[0]):
                print((x_+self.geometry[i][0])[0])
                self.data.append(go.Surface(x=x_r+self.geometry[i][0],
                                            y=y_r+self.geometry[i][1],
                                            z=z_r+self.geometry[i][2], 
                                            surfacecolor=color,
                                            colorscale=self.colorscale,
                                            showscale=False, opacity=0.9)
                                )
        else:
            # Todo: Use correct sizes of atoms
            showscale=True
            for j in range(Z.shape[0]):
                color = np.random.uniform(0, 0.1, self.geometry.shape[0])
                atomic_num = Z[j].numpy()
                surface_color = np.asarray(color + atomic_num / (len(self.atomtypes)-1))
                r = (atomic_num ** (1./root)) * size
                if r > 0:
                    """
                    atom = self.zToAtom(atomic_num)
                    name = atom.name
                    """
                    name = element_list[j]
                    x_r, y_r, z_r = x_*r, y_*r, z_*r
                    trace = go.Surface(x=x_r+self.geometry[j][0],
                                       y=y_r+self.geometry[j][1],
                                       z=z_r+self.geometry[j][2],
                                       surfacecolor=surface_color,
                                       cmin=0,
                                       cmax=1,
                                       showscale=False,
                                       name=name)
                    self.data.append(trace)

    def __plot__(self):
        self.fig = go.Figure(data=self.data) # , layout=self.layout
        plot(self.data,filename='test.html')
        self.fig.show()
        
    def __save__(self):
        plot = offline.plot(self.fig, filename="test.html")
    
    def get_elements(self, Z):
        elements = []
        Z = Z.flatten()
        for i in range(Z.size()[0]):
            elements.append(self.atomtypes[int(Z[i])])
        return elements
    
    def get_sphere(self):
        k = self.k
        theta = linspace(0,2*pi,k)
        phi = linspace(0,pi,k)
        x = outer(cos(theta),sin(phi))
        y = outer(sin(theta),sin(phi))
        z = outer(ones(k),cos(phi))  # note this is 2d now
        return x, y, z
    
    def zToAtom(self, z):
        return element(int(z))