""" 
    static 2D simulator for granular metamaterial
    based on Mark Shattuck and Jerry Zhang's code
    ver6: the top wall is pushed down
    new force-law by Sven
    input is a binary vector indicating stiff/soft particles
    output is force from the odd particles on the bottom wall
    
"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2025, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from scipy.special import lambertw as Wsf

class simulator:
    def __init__(self):
        pass   

    def evaluate(input):
        # input is a binary vector, 1: stiff 0: soft

        # find the indices of the stiff particles
        indices = np.nonzero(input)

        ## Experimental Parmeters
        Nx1 = 5
        Ny1 = 3
        Nx2 = 4
        Ny2 = 2

        N1 = Nx1 * Ny1
        N2 = Nx2 * Ny2
        N = Nx1 * Ny1 + Nx2 * Ny2 # total number of particles

        M = np.ones(N)   # Mass of particles
        Mw = 1. * Nx1  #mass of top wall
        Dn = np.ones(N)  # Diameter of particles
        Rn = 0.5 * Dn
        D = np.amax(Dn)  # maximum diameter

        kRatio = 3.0
        K_s = 1.0
        K = K_s * np.ones(N)
        K[indices] = K[indices] * kRatio

        alphacut = 1.0e-11 #force balance threshold

        Lx = Nx1 * D  #size of box
        Ly = ((Ny1 - 1) * math.sqrt(3) + 1) * D

        #TT=100    #total simulation time (short for demo).

        ## Physical Parameters
        g = 0

        ## Simulation Parmeters
        B = .5 #Drag coefficient
        dt = np.pi / np.sqrt(kRatio * K_s) / 10.0
        dt_half = 0.5 * dt
        dt_sq_half = 0.5 * dt**2
        B_rsc = 1. + B * dt_half

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
        x1 = np.transpose(x1)
        y1 = np.transpose(y1)
        x1 = np.flip(x1, 0)
        y1 = np.flip(y1, 0)
        x2 = np.transpose(x2)
        y2 = np.transpose(y2)
        x2 = np.flip(x2, 0)
        y2 = np.flip(y2, 0)

        x1 = np.reshape(x1, -1, order = 'F')
        x2 = np.reshape(x2, -1, order = 'F')
        y1 = np.reshape(y1, -1, order = 'F')
        y2 = np.reshape(y2, -1, order = 'F')
        x = np.concatenate((x1, x2), axis = 0)
        y = np.concatenate((y1, y2), axis = 0)
        x = np.reshape(x, N)
        y = np.reshape(y, N)

        vx = np.zeros(N)  # initial velocities of particles
        vy = np.zeros(N)
        vwy = 0.0
        ax = np.zeros(N)
        ay = np.zeros(N)
        ax_old = np.zeros(N)  # initial accelerations of particles
        ay_old = np.zeros(N)
        awy_old = 0.

        Fx = np.zeros(N)  # net force on each particle
        Fy = np.zeros(N)

        Fa = -0.01 * K_s

        ## Main Loop
        Rw = Lx #the walls are fixed at these positions
        Lw = 0.

        #Interaction detector and Force Law
        Fx[:] = 0.0
        Fy[:] = 0.0
        Fwy = Fa  # net force on top wall
        
        #interactions between particles
        for nn in range(N):
            for mm in range(nn + 1, N):
                dy = y[mm] - y[nn]
                Dnm = (Dn[nn] + Dn[mm]) / 2
                if np.abs(dy) < Dnm:
                    dx = x[mm] - x[nn]
                    if np.abs(dx) < Dnm:
                        dnm = np.sqrt(dx**2 + dy**2)
                        if dnm < Dnm:
                            d = Dnm - dnm
                            F = -K[nn] * K[mm] / (K[nn] + K[mm]) * np.sqrt(Rn[nn] * Rn[mm] / (Rn[nn] + Rn[mm])) * d**1.5 / dnm
                            dFx = F * dx
                            dFy = F * dy
                            Fx[nn] = Fx[nn] + dFx  #particle-particle Force Law
                            Fx[mm] = Fx[mm] - dFx
                            Fy[nn] = Fy[nn] + dFy  #particle-particle Force Law
                            Fy[mm] = Fy[mm] - dFy

        for nn in range(N):
            if x[nn] < Lw + Rn[nn]:
                Fx[nn] = Fx[nn] + K[nn] * np.sqrt(Rn[nn]) * (Lw + Rn[nn] - x[nn])**1.5
            elif x[nn] > Rw - Rn[nn]:
                Fx[nn] = Fx[nn] - K[nn] * np.sqrt(Rn[nn]) * (x[nn] - Rw + Rn[nn])**1.5
            if y[nn] < Rn[nn]:
                Fy[nn] = Fy[nn] + K[nn] * np.sqrt(Rn[nn]) * (Rn[nn] - y[nn])**1.5
            elif y[nn] > Ly - Rn[nn]:
                dFy = K[nn] * np.sqrt(Rn[nn]) * (y[nn] - Ly + Rn[nn])**1.5
                Fy[nn] = Fy[nn] - dFy
                Fwy = Fwy + dFy

        ax_old = Fx / M
        ay_old = Fy / M
        awy_old = Fwy / Mw

        # relaxation loop
        while True:
            x = x + vx * dt + ax_old * dt_sq_half  #first step in Verlet integration
            y = y + vy * dt + ay_old * dt_sq_half
            Ly = Ly + vwy * dt + awy_old * dt_sq_half
            
            #Interaction detector and Force Law
            Fx[:] = 0.0 # net force on each particle
            Fy[:] = 0.0
            Fwy = Fa  # net force on top wall
            
            #interactions between particles
            for nn in range(N):
                for mm in range(nn + 1, N):
                    dy = y[mm] - y[nn]
                    Dnm = (Dn[nn] + Dn[mm]) / 2
                    if np.abs(dy) < Dnm:
                        dx = x[mm] - x[nn]
                        if np.abs(dx) < Dnm:
                            dnm = np.sqrt(dx**2 + dy**2)
                            if dnm < Dnm:
                                d = Dnm - dnm
                                F = -K[nn] * K[mm] / (K[nn] + K[mm]) * np.sqrt(Rn[nn] * Rn[mm] / (Rn[nn] + Rn[mm])) * d**1.5 / dnm
                                dFx = F * dx
                                dFy = F * dy
                                Fx[nn] = Fx[nn] + dFx  #particle-particle Force Law
                                Fx[mm] = Fx[mm] - dFx
                                Fy[nn] = Fy[nn] + dFy  #particle-particle Force Law
                                Fy[mm] = Fy[mm] - dFy

            for nn in range(N):
                if x[nn] < Lw + Rn[nn]:
                    Fx[nn] = Fx[nn] + K[nn] * np.sqrt(Rn[nn]) * (Lw + Rn[nn] - x[nn])**1.5
                elif x[nn] > Rw - Rn[nn]:
                    Fx[nn] = Fx[nn] - K[nn] * np.sqrt(Rn[nn]) * (x[nn] - Rw + Rn[nn])**1.5
                if y[nn] < Rn[nn]:
                    Fy[nn] = Fy[nn] + K[nn] * np.sqrt(Rn[nn]) * (Rn[nn] - y[nn])**1.5
                elif y[nn] > Ly - Rn[nn]:
                    dFy = K[nn] * np.sqrt(Rn[nn]) * (y[nn] - Ly + Rn[nn])**1.5 
                    Fy[nn] = Fy[nn] - dFy
                    Fwy = Fwy + dFy

            if np.amax([np.amax(np.abs(Fx)), np.amax(np.abs(Fy)), np.abs(Fwy)]) < alphacut:
                break
            
            # correction for velocity dependent force
            ax = (Fx / M - B * (vx + ax_old * dt_half)) / B_rsc  
            ay = (Fy / M - B * (vy + ay_old * dt_half) - g) / B_rsc
            awy = (Fwy / Mw - B * (vwy + awy_old * dt_half) - g) / B_rsc
            
            vx = vx + (ax_old + ax) * dt_half  #second step in Verlet integration
            vy = vy + (ay_old + ay) * dt_half
            vwy = vwy + (awy_old + awy) * dt_half
            
            ax_old[:] = ax
            ay_old[:] = ay
            awy_old = awy
            
        # calculate forces on the walls from each particle
        Fwl = np.zeros(N)
        Fwr = np.zeros(N)
        Fwt = np.zeros(N)
        Fwb = np.zeros(N)

        ii = np.where(x< (Lw + Rn))
        Fwl[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Lw + Rn[ii] - x[ii])**1.5

        ii = np.where(y < Rn)
        Fwb[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Rn[ii] - y[ii])**1.5

        ii = np.where(x > (Rw - Rn))
        Fwr[ii] = K[ii] * np.sqrt(Rn[ii]) * (x[ii] - Rw + Rn[ii])**1.5

        ii = np.where(y > (Ly - Rn))
        Fwt[ii] = K[ii] * np.sqrt(Rn[ii]) * (y[ii] - Ly + Rn[ii])**1.5
        
        # returning the force O1 = F1*F3*F5
        return [(abs(Fwb[2]*Fwb[8]*Fwb[14]) ** (1./3)) / abs(Fa)]

    def showPacking(input, save=0):
        # input is a binary vector, 1: stiff 0: soft
        
        # find the indices of the stiff particles
        indices = np.nonzero(input)

        Nouts = [2, 8, 14] #output particles

        ## Experimental Parmeters
        Nx1=5
        Ny1=3
        Nx2=4
        Ny2=2

        N1=Nx1*Ny1
        N2=Nx2*Ny2
        N=Nx1*Ny1+Nx2*Ny2 # total number of particles
        K=np.ones(N)*10 # spring constant for harmonic force law
        M=np.ones(N)   # Mass of particles
        Dn=np.zeros(N)+1.0  # Diameter of particles
        D=np.amax(Dn)  # maximum diameter

        Ki=60
        kRatio = 3.0
        K=np.ones(N)*Ki
        K[indices] = K[indices]*kRatio
        
        A=D/100     #amplitude of displacement

        Lx=Nx1*D  #size of box
        Ly=((Ny1-1)*math.sqrt(3)+1)*D

        ep=-A/Ly #compresssion strain [Lx x]->[Lx x]*(1+ep)

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.001:D, D/2:Ly-D/2+0.001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.001:math.sqrt(3)*D]
        x1=np.transpose(x1)
        y1=np.transpose(y1)
        x1=np.flip(x1, 0)
        y1=np.flip(y1, 0)
        x2=np.transpose(x2)
        y2=np.transpose(y2)
        x2=np.flip(x2, 0)
        y2=np.flip(y2, 0)
        
        x1=np.reshape(x1, -1, order='F')
        x2=np.reshape(x2, -1, order='F')
        y1=np.reshape(y1, -1, order='F')
        y2=np.reshape(y2, -1, order='F')
        x=np.concatenate((x1, x2), axis=0)
        y=np.concatenate((y1, y2), axis=0)
        x=np.reshape(x, N)
        y=np.reshape(y, N)

        # Ly compression
        y=y*(1+ep)
        Ly=Ly*(1+ep)

        ## Main Loop
        
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
     
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Circle((x_now, y_now), D/2)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')
            e.set_zorder(0)
            if K[i]==Ki:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Circle((x_now, y_now), D/2)
            e.set_facecolor('none')
            e.set_zorder(0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
                e.set_alpha(1)
                e.set_zorder(10)
            ax.add_artist(e)
             
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black', zorder=20)
        plt.plot([0, 0], [0, Ly], color='black', zorder=20)
        plt.plot([0, Lx], [Ly, Ly], color='black', zorder=20)
        plt.plot([Lx, Lx], [Ly, 0], color='black', zorder=20)

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10, zorder=20)
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    
        if save == 1:
            plt.savefig('figures/configuration.png', bbox_inches='tight', format='png', dpi=600, transparent=True)
            plt.show()
            plt.close(fig)
        else:
            plt.show()

    def showPackingWithContacts(input, save=0):

        # input is a binary vector, 1: stiff 0: soft
        
        # find the indices of the stiff particles
        indices = np.nonzero(input)

        Nouts = [2, 8, 14] #output particles

        ## Experimental Parmeters
        Nx1 = 5
        Ny1 = 3
        Nx2 = 4
        Ny2 = 2

        N1 = Nx1 * Ny1
        N2 = Nx2 * Ny2
        N = Nx1 * Ny1 + Nx2 * Ny2 # total number of particles

        M = np.ones(N)   # Mass of particles
        Mw = 1. * Nx1  #mass of top wall
        Dn = np.ones(N)  # Diameter of particles
        Rn = 0.5 * Dn
        D = np.amax(Dn)  # maximum diameter

        kRatio = 3.0
        K_s = 1.0
        K = K_s * np.ones(N)
        K[indices] = K[indices] * kRatio

        alphacut = 1.0e-11 #force balance threshold

        Lx = Nx1 * D  #size of box
        Ly = ((Ny1 - 1) * math.sqrt(3) + 1) * D

        #TT=100    #total simulation time (short for demo).

        ## Physical Parameters
        g = 0

        ## Simulation Parmeters
        B = .5 #Drag coefficient
        dt = np.pi / np.sqrt(kRatio * K_s) / 10.0
        dt_half = 0.5 * dt
        dt_sq_half = 0.5 * dt**2
        B_rsc = 1. + B * dt_half

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
        x1 = np.transpose(x1)
        y1 = np.transpose(y1)
        x1 = np.flip(x1, 0)
        y1 = np.flip(y1, 0)
        x2 = np.transpose(x2)
        y2 = np.transpose(y2)
        x2 = np.flip(x2, 0)
        y2 = np.flip(y2, 0)

        x1 = np.reshape(x1, -1, order = 'F')
        x2 = np.reshape(x2, -1, order = 'F')
        y1 = np.reshape(y1, -1, order = 'F')
        y2 = np.reshape(y2, -1, order = 'F')
        x = np.concatenate((x1, x2), axis = 0)
        y = np.concatenate((y1, y2), axis = 0)
        x = np.reshape(x, N)
        y = np.reshape(y, N)

        vx = np.zeros(N)  # initial velocities of particles
        vy = np.zeros(N)
        vwy = 0.0
        ax = np.zeros(N)
        ay = np.zeros(N)
        ax_old = np.zeros(N)  # initial accelerations of particles
        ay_old = np.zeros(N)
        awy_old = 0.

        Fx = np.zeros(N)  # net force on each particle
        Fy = np.zeros(N)

        Fa = -0.01 * K_s

        ## Main Loop
        Rw = Lx #the walls are fixed at these positions
        Lw = 0.

        #Interaction detector and Force Law
        Fx[:] = 0.0
        Fy[:] = 0.0
        Fwy = Fa  # net force on top wall
        
        #interactions between particles
        for nn in range(N):
            for mm in range(nn + 1, N):
                dy = y[mm] - y[nn]
                Dnm = (Dn[nn] + Dn[mm]) / 2
                if np.abs(dy) < Dnm:
                    dx = x[mm] - x[nn]
                    if np.abs(dx) < Dnm:
                        dnm = np.sqrt(dx**2 + dy**2)
                        if dnm < Dnm:
                            d = Dnm - dnm
                            F = -K[nn] * K[mm] / (K[nn] + K[mm]) * np.sqrt(Rn[nn] * Rn[mm] / (Rn[nn] + Rn[mm])) * d**1.5 / dnm
                            dFx = F * dx
                            dFy = F * dy
                            Fx[nn] = Fx[nn] + dFx  #particle-particle Force Law
                            Fx[mm] = Fx[mm] - dFx
                            Fy[nn] = Fy[nn] + dFy  #particle-particle Force Law
                            Fy[mm] = Fy[mm] - dFy

        for nn in range(N):
            if x[nn] < Lw + Rn[nn]:
                Fx[nn] = Fx[nn] + K[nn] * np.sqrt(Rn[nn]) * (Lw + Rn[nn] - x[nn])**1.5
            elif x[nn] > Rw - Rn[nn]:
                Fx[nn] = Fx[nn] - K[nn] * np.sqrt(Rn[nn]) * (x[nn] - Rw + Rn[nn])**1.5
            if y[nn] < Rn[nn]:
                Fy[nn] = Fy[nn] + K[nn] * np.sqrt(Rn[nn]) * (Rn[nn] - y[nn])**1.5
            elif y[nn] > Ly - Rn[nn]:
                dFy = K[nn] * np.sqrt(Rn[nn]) * (y[nn] - Ly + Rn[nn])**1.5
                Fy[nn] = Fy[nn] - dFy
                Fwy = Fwy + dFy

        ax_old = Fx / M
        ay_old = Fy / M
        awy_old = Fwy / Mw

        # relaxation loop
        while True:
            x = x + vx * dt + ax_old * dt_sq_half  #first step in Verlet integration
            y = y + vy * dt + ay_old * dt_sq_half
            Ly = Ly + vwy * dt + awy_old * dt_sq_half
            
            #Interaction detector and Force Law
            Fx[:] = 0.0 # net force on each particle
            Fy[:] = 0.0
            Fwy = Fa  # net force on top wall
            
            #interactions between particles
            for nn in range(N):
                for mm in range(nn + 1, N):
                    dy = y[mm] - y[nn]
                    Dnm = (Dn[nn] + Dn[mm]) / 2
                    if np.abs(dy) < Dnm:
                        dx = x[mm] - x[nn]
                        if np.abs(dx) < Dnm:
                            dnm = np.sqrt(dx**2 + dy**2)
                            if dnm < Dnm:
                                d = Dnm - dnm
                                F = -K[nn] * K[mm] / (K[nn] + K[mm]) * np.sqrt(Rn[nn] * Rn[mm] / (Rn[nn] + Rn[mm])) * d**1.5 / dnm
                                dFx = F * dx
                                dFy = F * dy
                                Fx[nn] = Fx[nn] + dFx  #particle-particle Force Law
                                Fx[mm] = Fx[mm] - dFx
                                Fy[nn] = Fy[nn] + dFy  #particle-particle Force Law
                                Fy[mm] = Fy[mm] - dFy

            for nn in range(N):
                if x[nn] < Lw + Rn[nn]:
                    Fx[nn] = Fx[nn] + K[nn] * np.sqrt(Rn[nn]) * (Lw + Rn[nn] - x[nn])**1.5
                elif x[nn] > Rw - Rn[nn]:
                    Fx[nn] = Fx[nn] - K[nn] * np.sqrt(Rn[nn]) * (x[nn] - Rw + Rn[nn])**1.5
                if y[nn] < Rn[nn]:
                    Fy[nn] = Fy[nn] + K[nn] * np.sqrt(Rn[nn]) * (Rn[nn] - y[nn])**1.5
                elif y[nn] > Ly - Rn[nn]:
                    dFy = K[nn] * np.sqrt(Rn[nn]) * (y[nn] - Ly + Rn[nn])**1.5 
                    Fy[nn] = Fy[nn] - dFy
                    Fwy = Fwy + dFy

            if np.amax([np.amax(np.abs(Fx)), np.amax(np.abs(Fy)), np.abs(Fwy)]) < alphacut:
                break
            
            # correction for velocity dependent force
            ax = (Fx / M - B * (vx + ax_old * dt_half)) / B_rsc  
            ay = (Fy / M - B * (vy + ay_old * dt_half) - g) / B_rsc
            awy = (Fwy / Mw - B * (vwy + awy_old * dt_half) - g) / B_rsc
            
            vx = vx + (ax_old + ax) * dt_half  #second step in Verlet integration
            vy = vy + (ay_old + ay) * dt_half
            vwy = vwy + (awy_old + awy) * dt_half
            
            ax_old[:] = ax
            ay_old[:] = ay
            awy_old = awy
            
        # calculate forces on the walls from each particle
        Fwl = np.zeros(N)
        Fwr = np.zeros(N)
        Fwt = np.zeros(N)
        Fwb = np.zeros(N)

        ii = np.where(x< (Lw + Rn))
        Fwl[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Lw + Rn[ii] - x[ii])**1.5

        ii = np.where(y < Rn)
        Fwb[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Rn[ii] - y[ii])**1.5

        ii = np.where(x > (Rw - Rn))
        Fwr[ii] = K[ii] * np.sqrt(Rn[ii]) * (x[ii] - Rw + Rn[ii])**1.5

        ii = np.where(y > (Ly - Rn))
        Fwt[ii] = K[ii] * np.sqrt(Rn[ii]) * (y[ii] - Ly + Rn[ii])**1.5
        
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
     
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Circle((x_now, y_now), D/2)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')
            e.set_edgecolor('none')
            e.set_zorder(0)
            if K[i]==K_s:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Circle((x_now, y_now), D/2)
            e.set_facecolor('none')
            e.set_edgecolor('none')
            e.set_zorder(0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
                e.set_alpha(1)
                e.set_zorder(10)
            ax.add_artist(e)
             
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black', zorder=20)
        plt.plot([0, 0], [0, Ly], color='black', zorder=20)
        plt.plot([0, Lx], [Ly, Ly], color='black', zorder=20)
        plt.plot([Lx, Lx], [Ly, 0], color='black', zorder=20)

        # draw contacts
        for nn in range(N):
            for mm in range(nn + 1, N):
                dy = y[mm] - y[nn]
                Dnm = (Dn[nn] + Dn[mm]) / 2
                if np.abs(dy) < Dnm:
                    dx = x[mm] - x[nn]
                    if np.abs(dx) < Dnm:
                        dnm = np.sqrt(dx**2 + dy**2)
                        if dnm < Dnm:
                            d = Dnm - dnm
                            F = -K[nn] * K[mm] / (K[nn] + K[mm]) * np.sqrt(Rn[nn] * Rn[mm] / (Rn[nn] + Rn[mm])) * d**1.5 / dnm
                            plt.plot([x[mm], x[nn]],[y[mm], y[nn]], color='black', linewidth=3*abs(F)+1.5, zorder=20)
        
        # draw walls
        ii = np.where(x< (Lw + Rn))
        Fwl[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Lw + Rn[ii] - x[ii])**1.5
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]-Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwl[j])+0.5)
            
        ii = np.where(y < Rn)
        Fwb[ii] = -K[ii] * np.sqrt(Rn[ii]) * (Rn[ii] - y[ii])**1.5
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]], [y[j], y[j]-Dn[j]/2], color='black', linewidth=2.5*abs(Fwb[j])+0.5)

        ii = np.where(x > (Rw - Rn))
        Fwr[ii] = K[ii] * np.sqrt(Rn[ii]) * (x[ii] - Rw + Rn[ii])**1.5
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]+Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwr[j])+0.5)

        ii = np.where(y > (Ly - Rn))
        Fwt[ii] = K[ii] * np.sqrt(Rn[ii]) * (y[ii] - Ly + Rn[ii])**1.5
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]],[y[j], y[j]+Dn[j]/2], color='black', linewidth=2.5*abs(Fwt[j])+0.5)

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10, zorder=20)
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    
        if save == 1:
            plt.savefig('figures/C1.png', bbox_inches='tight', format='png', dpi=600, transparent=True)
            plt.show()
            plt.close(fig)
        else:
            plt.show()
        print(f"F1={round(abs(Fwb[2]), 5)}, F2={round(abs(Fwb[5]), 5)}, F3={round(abs(Fwb[8]), 5)},  F4={round(abs(Fwb[11]), 5)}, F5={round(abs(Fwb[14]), 5)}")
        return (abs(Fwb[2]*Fwb[8]*Fwb[14]) ** (1./3)) / abs(Fa)
