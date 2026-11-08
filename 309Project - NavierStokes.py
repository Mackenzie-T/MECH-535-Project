"""Navier Stokes solver

MECH 309 - Prof Mathias Legrand
Max Finch 261083266
Stephen Chang 261092274
Emma Sinasac 261008902
Russel Ismael 260985324"""


import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


#establish dimensions of cylinder
radius = 0.25* 128/math.pi
centre_x = 32
centre_y = 32


#define distance function as in project instructions
def d(x, y, r = radius, centre_x = centre_x, centre_y = centre_y):
    return(np.sqrt((x-centre_x)**2 + (y-centre_y)**2) - radius)

#define chi as in project instructions
def chi(x, y, epsilon):
  if (d(x, y) <= -epsilon):
     return(1)

  elif (d(x, y) >= epsilon):
     return(0)

  else:
      power = (-4*d(x,y)*epsilon) / ((d(x,y))**2 - epsilon**2)
      return( 1 / (1 + (math.e)**power))


  

def chi_plot(Lx, Ly, h):
    #plots chi 

    
    x_values = np.arange(0, Lx, h)
    y_values = np.arange(0, Ly, h)
    
    chi_values = np.zeros((int(Ly/h), (int(Lx/h))))
    x_count = 0
    y_count = 0
    
    #assign value of chi to each node in array
    for i in x_values:
        for j in y_values:
           
            chi_values[int(y_count)][int(x_count)] = chi(i,j,2*h)
            y_count += 1
        y_count = 0
        x_count += 1
    
    plt.contour(chi_values)
    plt.show()

#------------------------------------------------------------------------------------------

#evaluates dx on an array U, as in project description
def dx(U_array, Nx = 3, Ny = 3, h=1):
    
    Udx = np.zeros((U_array.shape))
    
    x_pos = 0
    for i in range(Nx): #in each column
        for j in range(Ny): #in each row
        
            if(i+1) > Nx - 1:
                x_pos = 0 #wrapping right
                Udx[j][i] = (U_array[j][x_pos] - U_array[j][i-1]) / (2*h)
            
            elif(i-1) < 0:
                x_pos = Nx - 1 #wrapping left
                Udx[j][i] = (U_array[j][i+1] - U_array[j][x_pos]) / (2*h)
            
            else:
                Udx[j][i] = (U_array[j][i+1] - U_array[j][i-1]) / (2*h)
    
    return(Udx)


#evaluates dy on an array U, as in project description
def dy(U_array, Nx = 3, Ny = 3, h=1):
    
    Udy = np.zeros((U_array.shape))
    
    y_pos = 0
    for i in range(Nx): #in each column
        for j in range(Ny): #in each row
        
            if(j+1) > Ny - 1:
                y_pos = 0 #wrapping down
                Udy[j][i] = (U_array[y_pos][i] - U_array[j-1][i]) / (2*h)
            
            elif(j-1) < 0:
                y_pos = Ny - 1 #wrapping up
                Udy[j][i] = (U_array[j+1][i] - U_array[y_pos][i]) / (2*h)
            
            else:
                Udy[j][i] = (U_array[j+1][i] - U_array[j-1][i]) / (2*h)

    return(Udy)


#solve laplacian with finite differences, as specified in project description
def laplace(U_array, Nx = 3, Ny = 3, h=1):
    
    U_laplace = np.zeros((U_array.shape))
    
    x_pos = 0
    y_pos = 0
    
    for i in range(Nx): #in each column
        for j in range(Ny): #in each row
            
            
            #I apologize, this is ugly, but it works
            if(i+1) > Nx - 1:
                x_pos = 0 #wrapping right
                
                if(j+1) > Ny - 1:
                    y_pos = 0 #wrapping down
                    U_laplace[j][i] = (U_array[j][x_pos] + U_array[j][i-1] + U_array[y_pos][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)
            
                elif(j-1) < 0:
                    y_pos = Ny - 1 #wrapping up
                
                    U_laplace[j][i] = (U_array[j][x_pos] + U_array[j][i-1] + U_array[j+1][i] + U_array[y_pos][i] - 4* U_array[j][i]) / (h**2)
                    
                else:
                    U_laplace[j][i] = (U_array[j][x_pos] + U_array[j][i-1] + U_array[j+1][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)
                    
            elif(i-1 < 0):
                x_pos = Nx - 1 #wrapping left
                
                if(j+1) > Ny - 1:
                    y_pos = 0 #wrapping down
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][x_pos] + U_array[y_pos][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)
            
                elif(j-1) < 0:
                    y_pos = Ny - 1 #wrapping up
                
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][x_pos] + U_array[j+1][i] + U_array[y_pos][i] - 4* U_array[j][i]) / (h**2)
                    
                else:
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][x_pos] + U_array[j+1][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)

                
                  
            else:
                
                if(j+1) > Ny - 1:
                    y_pos = 0 #wrapping down
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][i-1] + U_array[y_pos][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)
            
                elif(j-1) < 0:
                    y_pos = Ny - 1 #wrapping up
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][i-1] + U_array[j+1][i] + U_array[y_pos][i] - 4* U_array[j][i]) / (h**2)
                    
                else:
                    U_laplace[j][i] = (U_array[j][i+1] + U_array[j][i-1] + U_array[j+1][i] + U_array[j-1][i] - 4* U_array[j][i]) / (h**2)


            
    return(U_laplace)
    


def NavierStokes(Uinf, h, Lx = 64, Ly = 8, dt = 0.1, tolerance = 0.001):
    

    
    epsilon =  2*h
    Nx = int(Lx/h)
    Ny = int(Ly/h)
    
    n_of_time_steps = int(6/dt)
    
    nu = 10**-2
    eta = 10**-3
    
     #define chi everywhere       
    chi_mat = np.zeros((Ny, Nx), dtype = '2f')
    for i in range(Nx):
        for j in range(Ny):
           chi_mat[j][i] = chi(i, j, epsilon)
    
    #initialize uniform U (containing Ux and Uy at each point x and y) at time 0.
    U0 = np.zeros((Ny, Nx), dtype = '2f')
    for i in range(Nx): #in each column
        for j in range(Ny): #in each row
            U0[j][i][0] = Uinf
                

    U = U0
    U_collection = []
    U_collection.append(U0)
    P_collection = []
    V_collection = []
    
    U_mat_mult = np.zeros((Ny, Nx), dtype = '2f')
    
    mx = np.zeros((Ny, Nx))
    my = np.zeros((Ny, Nx))
    
    vx = np.zeros((Ny, Nx))
    vy = np.zeros((Ny, Nx))
    
    Ux = np.zeros((Ny, Nx))
    Uy = np.zeros((Ny, Nx))
    
    Ux_collection = []
    Uy_collection = []
    
    
    
    
    #time loop begins 
    for n in range(n_of_time_steps):
        
        #-------------------
        #U* computations
        #-------------------
        
        
        #compute derivatives
        Udx = dx(U, Nx, Ny, h)
        Udy = dy(U, Nx, Ny, h)
        
        for i in range(Nx):
            for j in range(Ny):
                chi_mat[j][i] = chi(i, j, epsilon)
        
        
        
        #Define matrix multiplicative factor (Jacobian) for use in Ustar eqn
        for i in range(Nx):
            for j in range(Ny):
        
                U_mat_mult[j][i] = np.matmul(np.dstack((Udx[j][i], Udy[j][i])), U[j][i])
        
        
        Ustar_guess = U
        
        
        #Find U* iteratively with fixed point
        Ustar  = np.zeros((Ny,Nx), dtype = '2f')
        
        while (np.any(abs(Ustar_guess - Ustar) > tolerance)):
            Ustar_guess = Ustar
            Ustar = ((-1*U_mat_mult) + (nu * laplace(Ustar_guess, Nx, Ny, h)) + U/dt) / ((1/dt) + (chi_mat/eta)) 
    
        #-------------------
        #Pstar computations
        #------------------
            
        #initialize variables
        Pstar = np.zeros((Ny, Nx))
        P_guess = Pstar + 1
        
        #compute Ustar derivatives
        Ustar_dx = dx(Ustar, Nx, Ny, h)
        Ustar_dy = dy(Ustar, Nx, Ny, h)
        
        #compute sum of x derivative x components and y derivative y components
        for i in range(Nx):
            for j in range(Ny):
                mx[j][i] = Ustar_dx[j][i][0] #x components
                my[j][i] = Ustar_dy[j][i][1] #y components
        P_factor = (mx + my) / dt
        
        #solve for Pstar using fixed point
        P_count  = 0
        while (np.any(abs(P_guess - Pstar) > tolerance)):
            P_guess = Pstar
            Pstar = ((laplace(P_guess, Nx, Ny, h) * (h**2)) + 4*P_guess - (P_factor* h**2)) /4
        
        
        P_collection.append(Pstar)
        
        #------------------
        #Un+1 computations
        #------------------
        
        
        #compute P derivatives
        Pderivatives = np.zeros((Ny, Nx), dtype = '2f')
        Pdx = dx(Pstar, Nx, Ny, h)
        Pdy = dy(Pstar, Nx, Ny, h)
        
        #set up derivative matrix
        for i in range(Nx):
            for j in range(Ny):
                Pderivatives[j][i][0] = Pdx[j][i]
                Pderivatives[j][i][1] = Pdy[j][i]
        
        #compute Un+1 field
        U_new = Ustar - (dt*Pderivatives)
        U_collection.append(U_new)
        
        
        #calculate vorticities
        for i in range(Nx):
            for j in range(Ny):
                vx[j][i] = U_new[j][i][1] #y components
                vy[j][i] = U_new[j][i][0] #x components
        
        V = vx - vy
        V_collection.append(V)
        
        #find Ux, Uy
        for i in range(Nx):
            for j in range(Ny):
                Ux[j][i] = U_new[j][i][0] #x components
                Uy[j][i] = U_new[j][i][1] #y components
                
        Ux_collection.append(Ux)
        Uy_collection.append(Uy)
        Ux = np.zeros((Ny, Nx))
        Uy = np.zeros((Ny, Nx))
        
        
        U = U_new
                
    
    #---------
    #Plotting
    #---------
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 6))
    
    #Add plot labels
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_aspect('equal')
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
        
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])

    #Add plot titles
    ax1.set_title("Pressure Contour")
    ax2.set_title("Vorticity Contour")
    ax3.set_title("X velocity Contour")
    ax4.set_title("Y velocity Contour")
    plt.subplots_adjust(hspace=1)  # Increase hspace for more vertical space between subplots

    
    
    # Initialize the contour plot with the first frame of data
    contour1 = ax1.contourf(P_collection[0])
    contour2 = ax2.contourf(V_collection[0])
    contour3 = ax3.contourf(Ux_collection[0])
    contour4 = ax4.contourf(Uy_collection[0])
    
    
    # Update function for the animation
    def update(frame):
        # Access the data for the current frame from your data_list
        data1 = P_collection[frame]
        data2 = V_collection[frame]
        data3 = Ux_collection[frame]
        data4 = Uy_collection[frame]
        
        # Redraw the updated contour plot with the new data
        ax1.contourf(data1)
        ax2.contourf(data2)
        ax3.contourf(data3)
        ax4.contourf(data4)
        
        return (ax1.collections + ax2.collections + ax3.collections + ax4.collections) # Return collections to update the plot

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(Ux_collection), interval=100, blit=True)
    ani.save('contours.gif', writer='imagemagick', fps=30)
    ani.save('contours.gif', writer='ffmpeg', fps=30)
    
    
    
    #calculates drag coefficient
    def DragCoef(Uinf, h, m, dt): # m is the time index

        Nx = int(Lx/h)
        Ny = int(Ly/h)

        rho = 1
        A = 0.5
        
        # Define summing term for quadrature scheme
        QuadSumDiff = 0 
        
        for i in range (Nx):
            for j in range (Ny):
                if m < n_of_time_steps:
                    QuadSumDiff += -1 * (1 - chi(i, j, 2*h)) * (U_collection[m+1][j][i][0] - U_collection[m][j][i][0]) * (math.pi/128)**2
               
        # Multiply sum by constants
        QuadSumDiff *= 2 / (rho * (Uinf**2) * A * dt)
        
        return QuadSumDiff
    
    #plots drag coeff calculated abovw
    def PlotDragCoef(Uinf, h, dt):
        
        Nx = int(Lx/h)
        Ny = int(Ly/h)

    
        DragCoef_collection = []
        times = []
        mf = int(n_of_time_steps/dt)
        t = 0
        
        for m in range(mf):
            
            DragCoef_collection.append(DragCoef(Uinf, h, m, dt))
            times.append(t*dt)
            t += dt
        
        #plots drag coeff
        plt.figure(figsize=(8,4))
        plt.plot(times, DragCoef_collection, label="C$_D$")
        plt.xlabel("Time [s]")
        plt.ylabel("Drag Coefficient, C$_D$")
        plt.legend()
        plt.show()
    
    PlotDragCoef(Uinf, h, dt)
    
    
    
    
                
    

NavierStokes(1, 1)

