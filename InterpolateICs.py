#Code to interpolate ICs onto a finer grid


#Load required libraries:
import numpy as np
from scipy.interpolate import griddata
from itertools import product
import matplotlib.pyplot as plt
import pdb 


#Program control:
#Model grid points are not exactly the same due to different bases:
AegirGrid = 1
meshFactor = 4
DedalusGrid = 0
GustoGrid = 0

#Interpolation options:
#method = 'nearest'
method = 'linear'	#seems to give best results.
#method = 'cubic'

MakePlot = 0
w2f = 1


#Read in ICs that you wish to interpolate:
randArrIn = np.loadtxt('./RandomSample_080_180_1.txt')


#Give input field correct parity for chosen basis:
randArrIn_flipx = np.flipud(randArrIn)
randArrIn = randArrIn + randArrIn_flipx
#plt.contourf(randArrIn)
#plt.show()
#pdb.set_trace()


#Convert 2D numpy array into 1D vector of values:
values = randArrIn.ravel()
#check ordering:
#print(randArrIn.shape)
#print(values[180:185])
#print(randArrIn[1,0:5])
#pdb.set_trace()
 

#Construct the grid associated with the original ICs:
Nx = 80
Nz = 180

Lx = 0.2
Lz = 0.45

dx = float(Lx)/Nx
dz = float(Lz)/(Nz-1)

x1 = np.arange(Nx)*dx
z1 = np.arange(Nz)*dz

points = np.array(list(product(x1, z1)))

#check ordering:
#print(points[0:5])
#print(points[180:185])
#print(x1[0:5])
#print(z1[0:5])

#check end points of domain for comparisons below:
#print(min(x1),max(x1))
#print(min(z1),max(z1))
#pdb.set_trace()

#check grid resolution:
#print(dx, dz)


#Define the new grid on which you wish to interpolate the ICs.
if AegirGrid == 1:
    Nx2 = Nx*meshFactor
    Nz2 = Nz*meshFactor
    Lx2 = max(x1)
    Lz2 = max(z1)
    dx2 = float(Lx2)/(Nx2-1)
    dz2 = float(Lz2)/(Nz2-1)
    x2 = np.arange(Nx2)*dx2
    z2 = np.arange(Nz2)*dz2

    #check endpoints of domain - should be same as original field:
    #print(min(x2),max(x2))
    #print(min(z2),max(z2))
    #print(x1)
    #print(x2)
    #print(z1)
    #print(z2)
    #pdb.set_trace()

    #check new grid resolution:
    #print(dx2, dz2)

if DedalusGrid == 1:
    x2 = np.loadtxt('./XGridDedalus.txt')
    z2 = np.loadtxt('./ZGridDedalus.txt')
if GustoGrid == 1: 
    x2 = 1
    z2 = 1

Xgrid2, Zgrid2 = np.meshgrid(x2,z2, indexing='ij')

#check ordering:
#print(Xgrid2.shape)

#check for interpolation points outside of domain of datums:
#print(np.min(Xgrid2),np.max(Xgrid2))
#print(np.min(Zgrid2),np.max(Zgrid2))
#pdb.set_trace()


#Interpolate ICs onto new mesh:
randArrOut = griddata(points, values, (Xgrid2, Zgrid2), method=method)
#check for nan points due to interpolation points outside of the domain of datums:
#print(np.max(randArrOut))
#pdb.set_trace()


#Plot results for comparison of ICs on coarse and fine grids:
if MakePlot == 1:

    #Set general plotting parameters assuming A4 page size:
    A4Width = 8.27
    MarginWidth = 1
    width = A4Width-2*MarginWidth
    height = 4
    #For scaling the A4 plot dimensions:
    ScaleFactor = 1
    #ScaleFactor = 0.7
    width = width*ScaleFactor
    height = height*ScaleFactor

    fig=plt.figure(figsize=(width,height))
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)

    ax1 = fig.add_subplot(grid[0,0])
    i1=ax1.contourf(randArrIn)

    ax2 = fig.add_subplot(grid[0,1])
    i2=ax2.contourf(randArrOut)

    plt.show()    

    plt.figure()
    plt.plot(x1, randArrIn[:,int(Nz/2.)],'g')
    plt.plot(x2, randArrOut[:,int(Nz2/2.)],'r')
    #plt.plot(x1, randArrIn[:,Nz-1],'b')
    #plt.plot(x2, randArrOut[:,Nz2-1],'c')
    plt.show()


if w2f == 1:
    if meshFactor==2: fnm_ICs = './RandomSample_160_360.txt'
    if meshFactor==4: fnm_ICs = './RandomSample_320_720.txt'
    if meshFactor==8: fnm_ICs = './RandomSample_640_1440.txt'
    np.savetxt(fnm_ICs,randArrOut)
