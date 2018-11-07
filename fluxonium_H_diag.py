#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:51:38 2017
In this script the Fluxonium Model is considered as defined in 
PhysRevB.87.024510
The circuit is a designed there, it can be seen as 
-an harmonic oscillator given by the L C elements with el and ec its energy 
-with a potential V(x) that in this specific sample V is given by the JJ and is cos term.
What we do we solve  numerically the diagonalization of the matrix H.O. + JJ

Noticed that in a paper  PhysRevLett.103.217004 they have an asymptotic expansion for matrix elements (of the Hamiltonian on the number operators N) but this is not the real limit of the fluzonium. So i don’t think is the way to do. Also because the theory do not give all the operator values

I the following the fluxonium is thought benign in the flux representation.
The chagrinning energy and inductance energy gives the H of the harmonic oscillator that i know how to write in the diagonal basis according to standards procedure. Than I add a potential in express on the basis of the harmonic oscillator. 
I may use the notation X brining the flux operator and V the number operators

I defined 3 classes:
-harmonicosci  it makes the matrix operator for H x position (flux) and v momenta (number of cooper paris) of the harmonic operators
-jjpiece_hobases it is just the matrix representation of the potentials
-htot  the matrix of the full systems
-diagonalform H tot diagonalised

the constructors has  argument 
		el, ec, ej , nelem 
that are respectively
inductance energy , chagrinning energy,  JJ energy harmonicosci
 and nelem the number of state where i cut the Hilber space. 
nelem =40  it is more than optimal I believe 30 is enough.

there is a lot of way to go at operative part but I guess you want the function 
ploten_cij_vij(el, ec, ej,        #flxonium parameter   \  
                       which_state=0     #states for which  you want to see mat element 
                       ,size_hilb=8 ):    # size of the truncated space

it call a diagonalised expression for many value of external flux in the loop from 0 to pi (periodicity pi) and it plot the following staff:
-energy level as function of the flux
-energy level- energy gun states  as function of the flux
-Matrix element of |< which_state| X| i >|=|< which_state| \hat phi| i > for i=0…numlineinplot
-Matrix element of |< which_state| V| i >|=|< which_state| \hat N| i > for i=0…numlineinplot
numlineinplot can be changed 


So if you run on some idle the file and you call the function

ploten_cij_vij(el, ec, ej,        #flxonium parameter   \  
                       which_state=0     #states for which  you want to see mat element 
                       ,size_hilb=30 )

and way to see the plots

The external flux is taken in unit of \Phi_0=h/(2e)  HERE THERE IS A 2PI??

@author: giovanni_viola
"""
import numpy.linalg
import math
import scipy.special

import matplotlib.pyplot 

def tf(n=1):
    return 10+n

# aux matrix elemenst
# x and p operators
def xopmm(m,n):
    '''matrix element for X'''
    if(n==m+1):
        krondelta=math.sqrt(n)
    elif(n==m-1):
        krondelta=math.sqrt(m)
    else:
        krondelta=0
    return krondelta/math.sqrt(2)
    
def vopmm(m,n):
    '''matrix element for v'''
    if(n==m+1):
        krondelta=math.sqrt(n)
    elif(n==m-1):
        krondelta=-math.sqrt(m)
    else:
        krondelta=0
    return krondelta/math.sqrt(2)


def mat_someparity(n,m,phi):
    """Gives the matrix element are given in B1 above PHYSICAL REVIEW B 87, 024510 """
    diff  = abs(n-m)/2
    n1    = min(n,m)
    funz  = scipy.special.genlaguerre(n1,2*diff)
    aux   = pow(-2.,-diff)*math.sqrt( math.factorial(n1)/( math.factorial(n1+2*diff))) *\
    math.pow(phi,2*diff)* math.exp(-phi**2/4)* funz(phi**2/2)
    #print(aux)
    return aux
        

def mat_diffparity(n,m,phi):
    """Gives the matrix element are given in B1 above PHYSICAL REVIEW B 87, 024510 """
    diff  = (abs(n-m)-1)/2
    n1    = min(n,m)
    funz  = scipy.special.genlaguerre(n1,2*diff+1)
    aux   = pow(-2.,-diff)*math.sqrt( math.factorial(n1)/\
                ( 2*math.factorial(n1+2*diff+1))) *\
            math.pow(phi,2*diff+1)* math.exp(-phi**2/4)* funz(phi**2/2)
    #print(aux)
    return aux


class harmonicosci: 
    ''' harmonic osci this construc the relevan matrix for the armonic oscilator H, X and H.
    those are express in the diagoanl basis for the H.
    - in the class i have also to define some internal variable.
    - to make a harmonic osci you do harmonicosci(EL,EC,EJ,number of element  the space is trocated)
    - .matriform the form of the H  .xop and .vop the result'''

    def __init__(self, elO, ecO,nelem0):
      # self.Sigmaen0 = abs(tclass)**2/(-eoclass+1j*deltaclass)
        self.el       =   elO
        self.ec       =   ecO
        self.ej       =   0

        self.nelem    =   nelem0
        self.phi_AO   =   math.pow(8*ecO/elO,0.25)
 

    def mmatele(self,n,m):
        '''matrix element for H'''
        if(n==m):
            return math.sqrt(8*self.el*self.ec)*n
        else:
            return 0
        return 

    
  # matrix form   for H X and V
    def matrixform_H(self):
        '''H matrix'''
        mat = [[self.mmatele(n,m) for n in range(self.nelem)] for m in range(self.nelem)] #numpy.zeros((self.nelem,self.nelem))
        return numpy.array(mat)
    
    def matrixform_X(self):
        '''X matrix'''
        xop = [[self.phi_AO*xopmm(n,m) for n in range(self.nelem)] for m in range(self.nelem)]
        return numpy.array(xop)
    
    def matrixform_V(self): 
        '''V matrix'''
        vop = [[vopmm(n,m)/self.phi_AO for n in range(self.nelem)] for m in range(self.nelem)]
        return numpy.array(vop)
    
    #def moop(self):
        #mat = [[self.mmatele(n,m) for n in range(self.nelem)] for m in range(self.nelem)]
    
class jjpiece_hobases:
    ''' interatcin term in h. osci bases '''

    def __init__(self, elO, ecO, ejO, nelem0):
      # self.Sigmaen0 = abs(tclass)**2/(-eoclass+1j*deltaclass)
        self.el       =   elO
        self.ec       =   ecO
        self.ej       =   ejO
        self.nelem    =   nelem0
        self.phi_AO   =   math.pow(8*ecO/elO,0.25)
        #self.phiext   =   externalflux #flux across the loop in uniti of phinot
        
    def mmatele(self,n,m,externalflux):
        if((n-m) % 2 ==0):
            value = math.cos(2*math.pi*externalflux)*mat_someparity(n,m,self.phi_AO)
        else:
            value =  math.sin(2*math.pi*externalflux)*mat_diffparity(n,m,self.phi_AO)
        return self.ej*value

    
    def matrixform_H(self,externalflux):
        mat = [[   self.mmatele(n,m,externalflux) for n in range(self.nelem)] for m in range(self.nelem)] #numpy.zeros((self.nelem,self.nelem))
        #print(mat)
        return numpy.array(mat)
    
#    def matrixform_X(self):
#        '''X matrix'''
#        xop = [[self.phi_AO*xopmm(n,m) for n in range(self.nelem)] for m in range(self.nelem)]
#        return numpy.array(xop)
#    
#    def matrixform_V(self): 
#        '''V matrix'''
#        vop = [[vopmm(n,m)/self.phi_AO for n in range(self.nelem)] for m in range(self.nelem)]
#        return numpy.array(vop)


class htot:
    ''' The full hamiltonin in the base of the HO and 
    than I move in the transformed based'''

    def __init__(self, elO, ecO, ejO, nelem0):
      # self.Sigmaen0 = abs(tclass)**2/(-eoclass+1j*deltaclass)
        self.el       =   elO
        self.ec       =   ecO
        self.ej       =   ejO
        self.nelem    =   nelem0
        self.phi_AO   =   math.pow(8*ecO/elO,0.25)
        #self.phiext   =   phiext

    def h0(self):
        aux    = harmonicosci(self.el,self.ec,self.nelem)
        aux2   = aux.matrixform_H()#,aux.matrixform_X(),aux.matrixform_V()]
        return aux2
    

    def h1(self,phiext ):
        aux    = jjpiece_hobases(self.el,self.ec,self.ej,self.nelem)
        aux2   = aux.matrixform_H(phiext)
        return aux2
        #return h1i
    
    def hfluxonium(self,phiext):
       
        return (self.h1(phiext))+(self.h0())
    
    def aval_avect(self,phiext):
        
        return numpy.linalg.eigh((self.hfluxonium(phiext))) 
        # here is important use eigh . eig do not make staff in order
    
    def matrixform_X(self):
        '''X matrix'''
        xop = [[self.phi_AO*xopmm(n,m) for n in range(self.nelem)] for m in range(self.nelem)]
        return numpy.array(xop)
    
    def matrixform_V(self): 
        '''V matrix'''
        vop = [[vopmm(n,m)/self.phi_AO for n in range(self.nelem)] for m in range(self.nelem)]
        return numpy.array(vop)
    



    
class diagonalform(htot):
        """diagonal form submodule
        """
        def __init__(self, elO, ecO, ejO, nelem0):
      # self.Sigmaen0 = abs(tclass)**2/(-eoclass+1j*deltaclass)
          self.el       =   elO
          self.ec       =   ecO
          self.ej       =   ejO
          self.nelem    =   nelem0
          self.phi_AO   =   math.pow(8*ecO/elO,0.25)
          self.htot     =   htot(elO, ecO, ejO, nelem0)
          
        
        
        def hxvdiag(self,phiex):
            MAT = numpy.transpose(self.htot.aval_avect(phiex)[1])
           # print(numpy.linalg.det(MAT))
            AUX =numpy.dot(MAT,self.htot.hfluxonium(phiex))
            hdiag = numpy.dot(AUX,numpy.transpose(MAT))
            
            AUX =numpy.dot(MAT,self.htot.matrixform_X())
            hdiagX = numpy.dot(AUX,numpy.transpose(MAT))
            AUX =numpy.dot(MAT,self.htot.matrixform_V())
            hdiagV = numpy.dot(AUX,numpy.transpose(MAT))
            return [hdiag,hdiagX,hdiagV]
        
        
        
        


def ploten_cij_vij(elO, ecO, ejO,        #flxonium parameter   \  
                       which_state=0     #states that you want to see mat eleemnt 
                       ,size_hilb=8):      #which state you cut the Hilbers stpace 50 is mor than enoght
    ''' this function take as imput the 3 parameter of the fluxonimu and split plot of En X and V vs phiext'''
    print('el=',elO,'ec=',ecO,'ej=',ejO,'which states=',which_state,'size cut H=',size_hilb )
    print(r'the Hamiltonina is peridic in $\varphi_{ext}$ with periodicity 1 in units Phi_0 ' )
    print('' )
    fluxdiaga=diagonalform(elO, ecO, ejO, size_hilb)
    densityflux=0.05 #density of point in flux range for the plot
    fluxmax=1 # max flux external in the plot
    fluxmin=0 # min flux external in the plot
    numlineinplot = 7 # how many line you want in the plot has to be < size_hilb
    # I have to understand a better way to make this plot
    xaxeslabelfluxext=r'$\varphi_{ext}\,[\Phi_0]$'
    
    
   
    phivec = numpy.arange(fluxmin,fluxmax,densityflux)
    auxx   = numpy.zeros(size_hilb)
    auxv   = numpy.zeros(size_hilb) 
    auxh   = numpy.zeros(size_hilb) 
    for ph in phivec:
    #v1=cccc2.aval_avect(ph)[0]
        [en , ve ,xe ] = fluxdiaga.hxvdiag(ph)
        auxh = numpy.c_[auxh,numpy.diagonal(en)]
        auxx = numpy.c_[auxx,ve[which_state,:]]
        auxv = numpy.c_[auxv,xe[which_state,:]]
    #TEST ax1.autoscale_view()
    #MAKE plot of egen values
    
    for index in range(numlineinplot):
        matplotlib.pyplot.plot(phivec, auxh[index,1:])
        matplotlib.pyplot.xlabel(xaxeslabelfluxext)
        matplotlib.pyplot.title('eigenenergies VS flux ')
        matplotlib.pyplot.ylabel(r'$E_n$')
    #v1=cccc2.aval_avect(ph)[0]
    #ax1.set_xlabel('phiext')
    #ax1.set_title('eenergy')
    matplotlib.pyplot.show()
    ax1=matplotlib.pyplot.show()
    
    #make the excitation energy and plot them
    for index in range(numlineinplot-1):
        matplotlib.pyplot.plot(phivec,-auxh[0,1:]+ auxh[index+1,1:])
        matplotlib.pyplot.title('excitation energy VS flux ')
        matplotlib.pyplot.xlabel(xaxeslabelfluxext)
        matplotlib.pyplot.ylabel(r'$E_n-E_0$')
        
     #v1=cccc2.aval_avect(ph)[0]
#    ax2.autoscale_view()
#    ax2.set_xlabel('phiext')
#    ax2.set_title('en-eo')
#    ax2.matplotlib.pyplot.show()
    matplotlib.pyplot.show()
    ax2=matplotlib.pyplot.show()
    
    
    #construct the matrix element of energy and plot them
    for index in range(numlineinplot):
        matplotlib.pyplot.plot(phivec, numpy.vectorize(abs)(auxx[index,1:]))
        matplotlib.pyplot.title('matrix elements of flux operator VS flux ')
        matplotlib.pyplot.xlabel(xaxeslabelfluxext)
        matplotlib.pyplot.ylabel(r'$|\langle i \vert \hat \phi \vert i \langle |$')
#    ax3.autoscale_view()
#    ax3.set_xlabel('phiext')
#    ax3.set_title('|\phi ij|')
    matplotlib.pyplot.show()
    ax3=matplotlib.pyplot.show()
    
    for index in range(numlineinplot):
        matplotlib.pyplot.plot(phivec, numpy.vectorize(abs)(auxv[index,1:]))
        matplotlib.pyplot.title('matrix elements of number operator VS flux ')
        matplotlib.pyplot.xlabel(xaxeslabelfluxext)
        matplotlib.pyplot.ylabel(r'$|\langle i \vert \hat N \vert i \langle |$')
#    ax5.autoscale_view()
#    ax4.set_xlabel('phiext')
#    ax4.set_title('|\phi ij|')
#    ax4.matplotlib.pyplot.show()
    matplotlib.pyplot.show()
    ax4=matplotlib.pyplot.show()
    
    #print("how can i make the 3 plot in a array of plot")
    #fig, axes =  matplotlib.pyplot.subplots(2, 2)
    #axes = ((ax1, ax2), (ax3, ax4)) 
    #axes.show()




#----------------------------------------------
    #----------------------------------------------
#----------------------------------------------
#----------------------------------------------
#    WORKING MATERIALS
    


def ploten(elO, ecO, ejO,        #flxonium parameter   \  
                       which_state=0     #states that you want to see mat eleemnt 
                       ,size_hilb=8):      #which state you cut the Hilbers stpace 50 is mor than enoght
    ''' this function take as imput the 3 parameter of the fluxonimu and split plot of En X and V vs phiext'''
    fluxdiaga=diagonalform(elO, ecO, ejO, size_hilb)
    densityflux=0.003 #density of point in flux range for the plot
    fluxmax=math.pi # max flux external in the plot
    fluxmin=0 # min flux external in the plot
    # I have to understand a better way to make this plot
    phivec = numpy.arange(fluxmin,fluxmax,densityflux)
    auxx   = numpy.zeros(size_hilb)
    auxv   = numpy.zeros(size_hilb) 
    auxh   = numpy.zeros(size_hilb) 
    for ph in phivec:
    #v1=cccc2.aval_avect(ph)[0]
        [en , ve ,xe ] = fluxdiaga.hxvdiag(ph)
        auxh = numpy.c_[auxh,numpy.diagonal(en)]
        auxx = numpy.c_[auxx,ve[0,:]]
        auxv = numpy.c_[auxv,xe[0,:]]
    for index in range(size_hilb):
        matplotlib.pyplot.plot(phivec, auxh[index,1:])
        matplotlib.pyplot.xlabel(xaxeslabelfluxext)
        matplotlib.pyplot.ylabel('En')
        
    #v1=cccc2.aval_avect(ph)[0]
#
#size=1 https://www.tutorialspoint.com/developers_best_practices/documentation_is_key.htm
#aaaa=harmonicosci(1,1,1,size)
#bbbb=jjpiece_hobases(1,1,1,size)
#tt=htot(1,1,1,size)
#
#
#print(bbbb.mmatele(1,1))
#print(tt.h1())
#
#print(tt.h1())
#print()
#print(tt.hfluxonium())
#print()
#print(tt.aval_avect())
#print(tt.diagonalform.xopdiag())


#size=10
##cccc2=htot(1,.1,.6,size)
##cccc2.hfluxonium(0)
#phivec = numpy.arange(0,math.pi*2,.091)
## aux will contain a lot of staff but to use append The firts place is fake
#aux=numpy.zeros(size) 
##aux=sorted(numpy.array(cccc2.aval_avect(0)[0] ))
#for ph in phivec:
#    #v1=cccc2.aval_avect(ph)[0]
#    aux=numpy.c_[aux,cccc2.aval_avect(ph)[0]]
#    #matplotlib.pyplot.plot(phivec,v1)
#
## to plot the egine values
#for index in range(size):
#    matplotlib.pyplot.plot(phivec, aux[index,1:])
#    #v1=cccc2.aval_avect(ph)[0]
#    
#    
#matplotlib.pyplot.show()
   

#ploten_cij_vij(1,1,1)    
#matplotlib.pyplot.show()
#
#    
#
#cccc3=diagonalform(1.,2.,2.6,size)
#
#
#auxx=numpy.zeros(size)
#auxv=numpy.zeros(size) 
#auxh=numpy.zeros(size) 
# 
##aux=sorted(numpy.array(cccc2.aval_avect(0)[0] ))
#for ph in phivec:
#    #v1=cccc2.aval_avect(ph)[0]
#    [en , ve ,xe ] = cccc3.hxvdiag(ph)
#    auxh = numpy.c_[auxh,numpy.diagonal(en)]
#    auxx = numpy.c_[auxx,ve[0,:]]
#    auxv = numpy.c_[auxv,xe[0,:]]
    #aux=numpy.c_[aux,cccc3.hxvdiag(ph)[0]]
    #matplotlib.pyplot.plot(phivec,v1)

#for index in range(size):
#    matplotlib.pyplot.plot(phivec, auxh[index,1:])
#    #v1=cccc2.aval_avect(ph)[0]
#    
#    
#matplotlib.pyplot.show()
#
#for index in range(size):
#    matplotlib.pyplot.plot(phivec, numpy.vectorize(abs)(auxx[index,1:]))
#    #v1=cccc2.aval_avect(ph)[0]
#    
#    
#matplotlib.pyplot.show()
#
#for index in range(size):
#    matplotlib.pyplot.plot(phivec, numpy.vectorize(abs)(auxv[index,1:]))
#    #v1=cccc2.aval_avect(ph)[0]
#    
#    
#matplotlib.pyplot.show()
