# static 2D simulator for granular metamaterial
# based on Mark Shattuck and Jerry Zhang's code
# ver6: the top wall is pushed down
# new force-law by Sven
# input is a binary vector indicating stiff/soft particles for two configurations
# the checkerboard setup
# output is force from the odd/even particles on the bottom wall

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import lambertw as Wsf

class simulator:
    def __init__(self):
        pass   

    def evaluate(input):
        # input is a binary vector, 1: stiff 0: soft

        indices = np.array(input)
        
        #### first config
        input1 = np.array(input[0:23])
        # find the indices of the stiff particles
        indices1 = np.nonzero(input1)

        ## Experimental Parmeters
        Nx1=5
        Ny1=3
        Nx2=4
        Ny2=2

        N=Nx1*Ny1+Nx2*Ny2 # total number of particles
        K=np.ones(N)*60 # spring constant for harmonic force law
        
        M=np.ones(N)   # Mass of particles
        Mw=1*Nx1  #mass of top wall
        Dn=np.zeros(N)+1.0  # Diameter of particles
        D=np.amax(Dn)  # maximum diameter
        L=10

        kRatio = 2.5
        K[indices1] = K[indices1]*kRatio
        V = 1/K
        Vw = np.amin(V)
        Kw=np.amax(K)
        
        B=.5 #Drag coefficient

        alphacut=1.0e-11 #force balance threshold

        Fa=-1;   #force applied to the top wall
        
        Lx=Nx1*D  #size of box
        Ly=((Ny1-1)*math.sqrt(3)+1)*D

        ## Physical Parameters
        g=0

        ## Simulation Parmeters
        dt=1e-2

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
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

        vx=np.zeros(N)  # initial velocities of particles
        vy=np.zeros(N)
        
        vwy=0   #initial velocity of top wall, top wall only moves in y

        ax_old=np.zeros(N)  # initial accelerations of particles
        ay_old=np.zeros(N)
        
        awy_old=0 #initial acceleration of top wall

        
        ## Main Loop
        Rw=Lx #the walls are fixed at these positions
        Lw=0

        # relaxation loop
        while np.amax([np.amax(ax_old),np.amax(ay_old),abs(awy_old)])>alphacut or np.amax([np.amax(ax_old),np.amax(ay_old),awy_old])==0:
        
            x = x+vx*dt+ax_old*dt**2/2  #first step in Verlet integration
            y = y+vy*dt+ay_old*dt**2/2

            Ly = Ly+vwy*dt+awy_old*dt**2/2
            
            #Interaction detector and Force Law
            Fx=np.zeros(N)  # net force on each particle
            Fy=np.zeros(N)
            Fwy=Fa;  # net force on top wall
            
            #interactions between particles
            for nn in range(1, N+1):
                for mm in range(nn+1, N+1):
                    dy=y[mm-1]-y[nn-1]
                    Dnm=(Dn[nn-1]+Dn[mm-1])/2
                    if(abs(dy) <= Dnm):
                        dx=x[mm-1]-x[nn-1]
                        dnm=dx**2+dy**2
                        if(dnm<Dnm**2):
                            dnm=np.sqrt(dnm)
                            d = (Dnm/dnm-1)
                            F = -np.real((4*L**3*np.exp(Wsf(-(d*D)/(4*np.exp(1)*L**2), -1)+1))/(D*(V[nn-1]+V[mm-1])))
                            Fx[nn-1]=Fx[nn-1]+F*dx  #particle-particle Force Law
                            Fx[mm-1]=Fx[mm-1]-F*dx
                            Fy[nn-1]=Fy[nn-1]+F*dy  #particle-particle Force Law
                            Fy[mm-1]=Fy[mm-1]-F*dy

            # interactions between particles and walls
            ii=np.where(x<(Lw+Dn/2))
            dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y<(Dn/2))
            dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            
            ii=np.where(x>(Rw-Dn/2))
            dw=2*(x[ii]-(Rw-Dn[ii]/2)) # Right wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y>(Ly-Dn/2))
            dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            Fwy=Fwy+np.sum(np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw))))
            
            # correction for velocity dependent force
            ax=(Fx/M-B*(vx+ax_old*dt/2))/(1+B*dt/2)  
            ay=(Fy/M-B*(vy+ay_old*dt/2)-g)/(1+B*dt/2)
            awy=(Fwy/Mw-B*(vwy+awy_old*dt/2)-g)/(1+B*dt/2)
            
            vx=vx+(ax_old+ax)*dt/2  #second step in Verlet integration
            vy=vy+(ay_old+ay)*dt/2
            vwy=vwy+(awy_old+awy)*dt/2
            
            ax_old=ax
            ay_old=ay
            awy_old=awy
            
        # calculate forces on the walls from each particle
        Fwl=np.zeros(N)
        Fwr=np.zeros(N)
        Fwt=np.zeros(N)
        Fwb=np.zeros(N)

        ii=np.where(x<(Lw+Dn/2))
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y<(Dn/2))
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(x>(Rw-Dn/2))
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y>(Ly-Dn/2))
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        #walloutput = Fwb[Fwb != 0]

        F1 = abs(Fwb[2]) * abs(Fwb[8]) * abs(Fwb[14])

        #### second config
        input2 = np.array(input[23:46])
        # find the indices of the stiff particles
        indices2 = np.nonzero(input2)

        ## Experimental Parmeters
        Nx1=5
        Ny1=3
        Nx2=4
        Ny2=2

        N=Nx1*Ny1+Nx2*Ny2 # total number of particles
        K=np.ones(N)*60 # spring constant for harmonic force law
        
        M=np.ones(N)   # Mass of particles
        Mw=1*Nx1  #mass of top wall
        Dn=np.zeros(N)+1.0  # Diameter of particles
        D=np.amax(Dn)  # maximum diameter
        L=10

        kRatio = 2.5
        K[indices2] = K[indices2]*kRatio
        V = 1/K
        Vw = np.amin(V)
        Kw=np.amax(K)
        
        B=.5 #Drag coefficient

        alphacut=1.0e-11 #force balance threshold

        Fa=-1;   #force applied to the top wall
        
        Lx=Nx1*D  #size of box
        Ly=((Ny1-1)*math.sqrt(3)+1)*D

        ## Physical Parameters
        g=0

        ## Simulation Parmeters
        dt=1e-2

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
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

        vx=np.zeros(N)  # initial velocities of particles
        vy=np.zeros(N)
        
        vwy=0   #initial velocity of top wall, top wall only moves in y

        ax_old=np.zeros(N)  # initial accelerations of particles
        ay_old=np.zeros(N)
        
        awy_old=0 #initial acceleration of top wall

        
        ## Main Loop
        Rw=Lx #the walls are fixed at these positions
        Lw=0

        # relaxation loop
        while np.amax([np.amax(ax_old),np.amax(ay_old),abs(awy_old)])>alphacut or np.amax([np.amax(ax_old),np.amax(ay_old),awy_old])==0:
        
            x = x+vx*dt+ax_old*dt**2/2  #first step in Verlet integration
            y = y+vy*dt+ay_old*dt**2/2

            Ly = Ly+vwy*dt+awy_old*dt**2/2
            
            #Interaction detector and Force Law
            Fx=np.zeros(N)  # net force on each particle
            Fy=np.zeros(N)
            Fwy=Fa;  # net force on top wall
            
            #interactions between particles
            for nn in range(1, N+1):
                for mm in range(nn+1, N+1):
                    dy=y[mm-1]-y[nn-1]
                    Dnm=(Dn[nn-1]+Dn[mm-1])/2
                    if(abs(dy) <= Dnm):
                        dx=x[mm-1]-x[nn-1]
                        dnm=dx**2+dy**2
                        if(dnm<Dnm**2):
                            dnm=np.sqrt(dnm)
                            d = (Dnm/dnm-1)
                            F = -np.real((4*L**3*np.exp(Wsf(-(d*D)/(4*np.exp(1)*L**2), -1)+1))/(D*(V[nn-1]+V[mm-1])))
                            Fx[nn-1]=Fx[nn-1]+F*dx  #particle-particle Force Law
                            Fx[mm-1]=Fx[mm-1]-F*dx
                            Fy[nn-1]=Fy[nn-1]+F*dy  #particle-particle Force Law
                            Fy[mm-1]=Fy[mm-1]-F*dy

            # interactions between particles and walls
            ii=np.where(x<(Lw+Dn/2))
            dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y<(Dn/2))
            dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            
            ii=np.where(x>(Rw-Dn/2))
            dw=2*(x[ii]-(Rw-Dn[ii]/2)) # Right wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y>(Ly-Dn/2))
            dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            Fwy=Fwy+np.sum(np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw))))
            
            # correction for velocity dependent force
            ax=(Fx/M-B*(vx+ax_old*dt/2))/(1+B*dt/2)  
            ay=(Fy/M-B*(vy+ay_old*dt/2)-g)/(1+B*dt/2)
            awy=(Fwy/Mw-B*(vwy+awy_old*dt/2)-g)/(1+B*dt/2)
            
            vx=vx+(ax_old+ax)*dt/2  #second step in Verlet integration
            vy=vy+(ay_old+ay)*dt/2
            vwy=vwy+(awy_old+awy)*dt/2
            
            ax_old=ax
            ay_old=ay
            awy_old=awy
            
        # calculate forces on the walls from each particle
        Fwl=np.zeros(N)
        Fwr=np.zeros(N)
        Fwt=np.zeros(N)
        Fwb=np.zeros(N)

        ii=np.where(x<(Lw+Dn/2))
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y<(Dn/2))
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(x>(Rw-Dn/2))
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y>(Ly-Dn/2))
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        #walloutput = Fwb[Fwb != 0]

        F2 = abs(Fwb[5]) * abs(Fwb[11])

        overlap = np.sum(np.abs(indices[0:23]-indices[23:46]))
        softParticles = (1+np.sum((1-indices[0:23]))) * (1+np.sum((1-indices[23:46])))

        return round(F1, 5), round(F2, 5), overlap, softParticles

    def showPacking(input, save=0):
        # input is a binary vector, 1: stiff 0: soft
        
        #### first config
        input1 = np.array(input[0:23])
        # find the indices of the stiff particles
        indices1 = np.nonzero(input1)

        Nouts = [2, 8, 14]

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

        kRatio = 2.5
        K[indices1] = K[indices1]*kRatio
        
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
        ells = []
        m_all = []
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                       
            e = Ellipse((x_now, y_now), D,D,0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
            ells.append(e)
            if K[i]==10:
                m_all.append(1)
            else:
                m_all.append(10)


        i = 0
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')

            if m_all[i] == 1:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

            i += 1
                    
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black')
        plt.plot([0, 0], [0, Ly], color='black')
        plt.plot([0, Lx], [Ly, Ly], color='black')
        plt.plot([Lx, Lx], [Ly, 0], color='black')

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10)

        plt.show() 
        if save == 1:
            fig.savefig(dpi = 300)

        #### second config
        input2 = np.array(input[23:46])
        # find the indices of the stiff particles
        indices2 = np.nonzero(input2)

        Nouts = [5, 11]

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

        kRatio = 2.5
        K[indices2] = K[indices2]*kRatio
        
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
        ells = []
        m_all = []
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                       
            e = Ellipse((x_now, y_now), D,D,0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
            ells.append(e)
            if K[i]==10:
                m_all.append(1)
            else:
                m_all.append(10)


        i = 0
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')

            if m_all[i] == 1:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

            i += 1
                    
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black')
        plt.plot([0, 0], [0, Ly], color='black')
        plt.plot([0, Lx], [Ly, Ly], color='black')
        plt.plot([Lx, Lx], [Ly, 0], color='black')

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10)

        plt.show() 
        if save == 1:
            fig.savefig(dpi = 300)

    def showPackingWithContacts(input, save=0):

        # input is a binary vector, 1: stiff 0: soft
        indices = np.array(input)
        
        #### first config
        input1 = np.array(input[0:23])
        # find the indices of the stiff particles
        indices1 = np.nonzero(input1)

        Nouts=[2, 8, 14] #output particle

        ## Experimental Parmeters
        Nx1=5
        Ny1=3
        Nx2=4
        Ny2=2
        Fa = -1
        N=Nx1*Ny1+Nx2*Ny2 # total number of particles
        
        M=np.ones(N)   # Mass of particles
        Mw=1.0*Nx1  #mass of top wall
        Dn=np.zeros(N)+1.0  # Diameter of particles
        D=np.amax(Dn)  # maximum diameter
        L=10
        
        Ki=60
        kRatio = 2.5
        K=np.ones(N)*Ki
        K[indices1] = K[indices1]*kRatio
        V = 1/K
        Vw = np.amin(V)
        
        B=.5 #Drag coefficient

        alphacut=1.0e-11 #force balance threshold
        
        Lx=Nx1*D  #size of box
        Ly=((Ny1-1)*math.sqrt(3)+1)*D

        ## Physical Parameters
        g=0

        ## Simulation Parmeters
        dt=1e-2

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
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

        vx=np.zeros(N)  # initial velocities of particles
        vy=np.zeros(N)
        
        vwy=0   #initial velocity of top wall, top wall only moves in y

        ax_old=np.zeros(N)  # initial accelerations of particles
        ay_old=np.zeros(N)
        
        awy_old=0 #initial acceleration of top wall

        
        ## Main Loop
        Rw=Lx #the walls are fixed at these positions
        Lw=0

        # relaxation loop
        while np.amax([np.amax(ax_old),np.amax(ay_old),abs(awy_old)])>alphacut or np.amax([np.amax(ax_old),np.amax(ay_old),awy_old])==0:
        
            x = x+vx*dt+ax_old*dt**2/2  #first step in Verlet integration
            y = y+vy*dt+ay_old*dt**2/2
            Ly = Ly+vwy*dt+awy_old*dt**2/2
            
            #Interaction detector and Force Law
            Fx=np.zeros(N)  # net force on each particle
            Fy=np.zeros(N)
            Fwy=Fa;  # net force on top wall
            
            #interactions between particles
            for nn in range(1, N+1):
                for mm in range(nn+1, N+1):
                    dy=y[mm-1]-y[nn-1]
                    Dnm=(Dn[nn-1]+Dn[mm-1])/2
                    if(abs(dy) <= Dnm):
                        dx=x[mm-1]-x[nn-1]
                        dnm=dx**2+dy**2
                        if(dnm<Dnm**2):
                            dnm=np.sqrt(dnm)
                            d = (Dnm/dnm-1)
                            F = -np.real((4*L**3*np.exp(Wsf(-(d*D)/(4*np.exp(1)*L**2), -1)+1))/(D*(V[nn-1]+V[mm-1])))
                            Fx[nn-1]=Fx[nn-1]+F*dx  #particle-particle Force Law
                            Fx[mm-1]=Fx[mm-1]-F*dx
                            Fy[nn-1]=Fy[nn-1]+F*dy  #particle-particle Force Law
                            Fy[mm-1]=Fy[mm-1]-F*dy

            # interactions between particles and walls
            ii=np.where(x<(Lw+Dn/2))
            dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y<(Dn/2))
            dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
           
            ii=np.where(x>(Rw-Dn/2))
            dw=2*(x[ii]-(Rw-Dn[ii]/2)) # Right wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y>(Ly-Dn/2))
            dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            Fwy=Fwy+np.sum(np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw))))
            
            # correction for velocity dependent force
            ax=(Fx/M-B*(vx+ax_old*dt/2))/(1+B*dt/2)  
            ay=(Fy/M-B*(vy+ay_old*dt/2)-g)/(1+B*dt/2)
            awy=(Fwy/Mw-B*(vwy+awy_old*dt/2)-g)/(1+B*dt/2)
            
            vx=vx+(ax_old+ax)*dt/2  #second step in Verlet integration
            vy=vy+(ay_old+ay)*dt/2
            vwy=vwy+(awy_old+awy)*dt/2
            
            ax_old=ax
            ay_old=ay
            awy_old=awy
            
        # calculate forces on the walls from each particle
        Fwl=np.zeros(N)
        Fwr=np.zeros(N)
        Fwt=np.zeros(N)
        Fwb=np.zeros(N)

        ii=np.where(x<(Lw+Dn/2))
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y<(Dn/2))
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(x>(Rw-Dn/2))
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y>(Ly-Dn/2))
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        ells = []
        m_all = []
        
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Ellipse((x_now, y_now), D,D,0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
            ells.append(e)
            if K[i]==Ki:
                m_all.append(1)
            else:
                m_all.append(10)


        i = 0
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')

            if m_all[i] == 1:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

            i += 1
                    
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black')
        plt.plot([0, 0], [0, Ly], color='black')
        plt.plot([0, Lx], [Ly, Ly], color='black')
        plt.plot([Lx, Lx], [Ly, 0], color='black')

        # draw contacts
        for nn in range(1, N+1):
            for mm in range(nn+1, N+1):
                dy=y[mm-1]-y[nn-1]
                Dnm=(Dn[nn-1]+Dn[mm-1])/2
                if (abs(dy)<=Dnm):
                    dx=x[mm-1]-x[nn-1]
                    dnm=dx**2+dy**2
                    if (dnm<Dnm**2):
                        dnm=np.sqrt(dnm)
                        F=-2*K[nn-1]*K[mm-1]/(K[nn-1]+K[mm-1])*(Dnm/dnm-1)
                        plt.plot([x[mm-1], x[nn-1]],[y[mm-1], y[nn-1]], color='black', linewidth=2.5*abs(F)+0.5)
        
        # draw walls
        ii=np.where(x<Lw+Dn/2)
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]-Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwl[j])+0.5)
            
        ii=np.where(y<Dn/2)
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]], [y[j], y[j]-Dn[j]/2], color='black', linewidth=2.5*abs(Fwb[j])+0.5)

        ii=np.where(x>Rw-Dn/2)
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]+Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwr[j])+0.5)

        ii=np.where(y>Ly-Dn/2)
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]],[y[j], y[j]+Dn[j]/2], color='black', linewidth=2.5*abs(Fwt[j])+0.5)

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10)

        plt.show()

        F1 = abs(Fwb[2]) * abs(Fwb[8]) * abs(Fwb[14])
        print("C1:")
        print("f1="+"{:.2f}".format(abs(Fwb[2]))+", f2="+"{:.2f}".format(abs(Fwb[8]))+", f3="+"{:.2f}".format(abs(Fwb[14])))
        print("soft particles: " + str(np.sum((1-indices[0:23]))))

        #### second config
        input2 = np.array(input[23:46])
        # find the indices of the stiff particles
        indices2 = np.nonzero(input2)

        Nouts=[5, 11] #output particle

        ## Experimental Parmeters
        Nx1=5
        Ny1=3
        Nx2=4
        Ny2=2
        Fa = -1
        N=Nx1*Ny1+Nx2*Ny2 # total number of particles
        
        M=np.ones(N)   # Mass of particles
        Mw=1.0*Nx1  #mass of top wall
        Dn=np.zeros(N)+1.0  # Diameter of particles
        D=np.amax(Dn)  # maximum diameter
        L=10
        
        Ki=60
        kRatio = 2.5
        K=np.ones(N)*Ki
        K[indices2] = K[indices2]*kRatio
        V = 1/K
        Vw = np.amin(V)
        
        B=.5 #Drag coefficient

        alphacut=1.0e-11 #force balance threshold
        
        Lx=Nx1*D  #size of box
        Ly=((Ny1-1)*math.sqrt(3)+1)*D

        ## Physical Parameters
        g=0

        ## Simulation Parmeters
        dt=1e-2

        ## Initial Conditions: # position the particles in rows and columns

        [x1, y1] = np.mgrid[D/2:Lx-D/2+0.0001:D, D/2:Ly-D/2+0.0001:math.sqrt(3)*D]
        [x2, y2] = np.mgrid[D:Lx-D+0.0001:D, (math.sqrt(3)+1)*D/2:Ly-(math.sqrt(3)+1)*D/2+0.0001:math.sqrt(3)*D]
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

        vx=np.zeros(N)  # initial velocities of particles
        vy=np.zeros(N)
        
        vwy=0   #initial velocity of top wall, top wall only moves in y

        ax_old=np.zeros(N)  # initial accelerations of particles
        ay_old=np.zeros(N)
        
        awy_old=0 #initial acceleration of top wall

        
        ## Main Loop
        Rw=Lx #the walls are fixed at these positions
        Lw=0

        # relaxation loop
        while np.amax([np.amax(ax_old),np.amax(ay_old),abs(awy_old)])>alphacut or np.amax([np.amax(ax_old),np.amax(ay_old),awy_old])==0:
        
            x = x+vx*dt+ax_old*dt**2/2  #first step in Verlet integration
            y = y+vy*dt+ay_old*dt**2/2
            Ly = Ly+vwy*dt+awy_old*dt**2/2
            
            #Interaction detector and Force Law
            Fx=np.zeros(N)  # net force on each particle
            Fy=np.zeros(N)
            Fwy=Fa;  # net force on top wall
            
            #interactions between particles
            for nn in range(1, N+1):
                for mm in range(nn+1, N+1):
                    dy=y[mm-1]-y[nn-1]
                    Dnm=(Dn[nn-1]+Dn[mm-1])/2
                    if(abs(dy) <= Dnm):
                        dx=x[mm-1]-x[nn-1]
                        dnm=dx**2+dy**2
                        if(dnm<Dnm**2):
                            dnm=np.sqrt(dnm)
                            d = (Dnm/dnm-1)
                            F = -np.real((4*L**3*np.exp(Wsf(-(d*D)/(4*np.exp(1)*L**2), -1)+1))/(D*(V[nn-1]+V[mm-1])))
                            Fx[nn-1]=Fx[nn-1]+F*dx  #particle-particle Force Law
                            Fx[mm-1]=Fx[mm-1]-F*dx
                            Fy[nn-1]=Fy[nn-1]+F*dy  #particle-particle Force Law
                            Fy[mm-1]=Fy[mm-1]-F*dy

            # interactions between particles and walls
            ii=np.where(x<(Lw+Dn/2))
            dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y<(Dn/2))
            dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
           
            ii=np.where(x>(Rw-Dn/2))
            dw=2*(x[ii]-(Rw-Dn[ii]/2)) # Right wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fx[ii] = Fx[ii] - deltaF
            
            ii=np.where(y>(Ly-Dn/2))
            dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
            deltaF = np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
            Fy[ii] = Fy[ii] - deltaF
            Fwy=Fwy+np.sum(np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw))))
            
            # correction for velocity dependent force
            ax=(Fx/M-B*(vx+ax_old*dt/2))/(1+B*dt/2)  
            ay=(Fy/M-B*(vy+ay_old*dt/2)-g)/(1+B*dt/2)
            awy=(Fwy/Mw-B*(vwy+awy_old*dt/2)-g)/(1+B*dt/2)
            
            vx=vx+(ax_old+ax)*dt/2  #second step in Verlet integration
            vy=vy+(ay_old+ay)*dt/2
            vwy=vwy+(awy_old+awy)*dt/2
            
            ax_old=ax
            ay_old=ay
            awy_old=awy
            
        # calculate forces on the walls from each particle
        Fwl=np.zeros(N)
        Fwr=np.zeros(N)
        Fwt=np.zeros(N)
        Fwb=np.zeros(N)

        ii=np.where(x<(Lw+Dn/2))
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y<(Dn/2))
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(x>(Rw-Dn/2))
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))

        ii=np.where(y>(Ly-Dn/2))
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        ells = []
        m_all = []
        
        for i in range(N):
            x_now = x[i]#%Lx
            y_now = y[i]#%Ly                    
            e = Ellipse((x_now, y_now), D,D,0)
            if i in Nouts:
                e.set_edgecolor((0, 0, 1))
                e.set_linewidth(4)
            ells.append(e)
            if K[i]==Ki:
                m_all.append(1)
            else:
                m_all.append(10)


        i = 0
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('C2')

            if m_all[i] == 1:
                e.set_alpha(0.4)
            else:
                e.set_alpha(0.8)

            i += 1
                    
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

        # draw walls
        plt.plot([0, Lx], [0, 0], color='black')
        plt.plot([0, 0], [0, Ly], color='black')
        plt.plot([0, Lx], [Ly, Ly], color='black')
        plt.plot([Lx, Lx], [Ly, 0], color='black')

        # draw contacts
        for nn in range(1, N+1):
            for mm in range(nn+1, N+1):
                dy=y[mm-1]-y[nn-1]
                Dnm=(Dn[nn-1]+Dn[mm-1])/2
                if (abs(dy)<=Dnm):
                    dx=x[mm-1]-x[nn-1]
                    dnm=dx**2+dy**2
                    if (dnm<Dnm**2):
                        dnm=np.sqrt(dnm)
                        F=-2*K[nn-1]*K[mm-1]/(K[nn-1]+K[mm-1])*(Dnm/dnm-1)
                        plt.plot([x[mm-1], x[nn-1]],[y[mm-1], y[nn-1]], color='black', linewidth=2.5*abs(F)+0.5)
        
        # draw walls
        ii=np.where(x<Lw+Dn/2)
        dw=2*(x[ii]-Lw-Dn[ii]/2) #Left wall
        Fwl[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]-Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwl[j])+0.5)
            
        ii=np.where(y<Dn/2)
        dw=2*(y[ii]-Dn[ii]/2)  #Bottom wall
        Fwb[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]], [y[j], y[j]-Dn[j]/2], color='black', linewidth=2.5*abs(Fwb[j])+0.5)

        ii=np.where(x>Rw-Dn/2)
        dw=2*(x[ii]-(Rw-Dn[ii]/2))  #Right wall
        Fwr[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]+Dn[j]/2], [y[j], y[j]], color='black', linewidth=2.5*abs(Fwr[j])+0.5)

        ii=np.where(y>Ly-Dn/2)
        dw=2*(y[ii]-(Ly-Dn[ii]/2))  #Top wall
        Fwt[ii]=-np.real((2*L**3*np.exp(Wsf(-(dw*D)/(2*np.exp(1)*L**2), -1)+1))/(D*(V[ii]+Vw)))
        for i in range(0, len(ii)):
            for j in ii[i]:
                plt.plot([x[j], x[j]],[y[j], y[j]+Dn[j]/2], color='black', linewidth=2.5*abs(Fwt[j])+0.5)

        plt.plot([0, Lx],[Ly, Ly], color='red', linewidth=10)

        plt.show()

        F2 = abs(Fwb[5]) * abs(Fwb[11])

        print("C2:")
        print("f1="+"{:.2f}".format(abs(Fwb[5]))+", f2="+"{:.2f}".format(abs(Fwb[11])))
        print("soft particles: " + str(np.sum((1-indices[23:46]))))

        overlap = np.sum(np.abs(indices[0:23]-indices[23:46]))
        softParticles = (1+np.sum((1-indices[0:23]))) * (1+np.sum((1-indices[23:46])))

        return round(F1, 5), round(F2, 5), overlap, softParticles