import numpy as np
import matplotlib.pyplot as plt

del_x = 0.05
del_y = 0.05
beta = del_x / del_y
k = 50  #thermal conductivity
rows = int((2/del_y)) + 1
columns = int((1/del_x)) + 1
heat_transfer = []

def solve_and_plot(scheme):
    #initial temperature of all points except the bottom line are set to 30 deg.C for faster convergence
    temp_old = np.zeros([rows, columns])
    temp_new = np.zeros([rows, columns])
    for i in range(rows):
        for j in range(columns):
            temp_new[i][j] = 30
    temp_new[-1, :] = 100  # Bottom boundary
    temp_old = temp_new.copy()  # Initialize temp_old with boundary conditions

    error = 1 #dummy error to get into the loop
    step = 0
    
    print("Solving using the", scheme.__name__ ,"scheme:\n")
    #Iterate
    while (error >= 0.01):
        error = 0
        scheme(temp_old, temp_new)
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                error += abs(temp_new[i][j] - temp_old[i][j])
                temp_old[i][j] = temp_new[i][j]
        step += 1
        #Plot temperature contour
        X = np.arange(0,columns)
        Y = np.flip(np.arange(0,rows))
        plt.contourf(X,Y,temp_new, cmap = 'gist_heat')
        plt.colorbar(label = 'Temperature')
        plt.xlabel("X gridpoints --->")
        plt.ylabel("Y gridpoints --->")
        plt.title("Final temperature distribution: ") 
        plt.show()
    print("Number of iterations taken to converge: ",step)
    #Calculate heat transfer
    for i in range(columns):
        heat_transfer.append(50.0*(temp_new[rows-1][i] - temp_new[rows-2][i]) / del_y)
    print("Heat transfer along the bottom boundary: ",heat_transfer)
    #plot heat transfer
    plt.title("Heat Transfer along the bottom Boundary")
    plt.plot(heat_transfer)
    plt.grid(True)
    plt.xlabel("X gridpoints-->")
    plt.ylabel("Magnitude of Heat transfer (in W/m^2)-->")
    plt.show()

#Function for thomas algorithm
def tdma_solve(a,b,c,d):
    n = len(a)
    bprime = b.copy()
    dprime = d.copy()
    for i in range(1,n+1):
        factor = a[i-1] / bprime[i-1]
        bprime[i] -= factor*c[i-1]
        dprime[i] -= factor*dprime[i-1]
    
    nx = len(d)
    x = np.zeros(nx)
    x[-1] = dprime[-1] / bprime[-1]
    for i in range(nx-2,-1,-1):
        x[i] = (dprime[i] - c[i]*x[i+1]) / bprime[i]
    
    return x

#Point gauss-seidel method:
def Point_Gauss_Seidel(temp_old, temp_new):
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            temp_new[i][j] = (0.5 / (1 + beta**2))*(temp_new[i-1][j] + temp_old[i+1][j] + (beta**2)*temp_old[i][j+1] + (beta**2)*temp_new[i][j-1])

#Point Successive Over Relaxation:
def PSOR(temp_old, temp_new):
    omega = 1.79
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            temp_new[i][j] = (1 - omega)*temp_old[i][j] + ((0.5*omega) / (1 + beta**2))*(temp_new[i-1][j] + temp_old[i+1][j] + (beta**2)*temp_old[i][j+1] + (beta**2)*temp_new[i][j-1])

#Line gauss-seidel method:
def Line_Gauss_Seidel(temp_old, temp_new):
    for j in range(1,columns-1):
        a = np.ones(rows-3) * -1.0
        b = np.ones(rows-2) * 2 * (1.0 + beta**2)
        c = np.ones(rows-3) * -1.0
        d = np.zeros(rows-2)
        for i in range(1,rows-1):
            d[i-1] = beta**2 * (temp_new[i][j-1] + temp_old[i][j+1])
        d[0] += 30
        d[-1] += 100

        new_values = tdma_solve(a,b,c,d)
        temp_new[1:rows-1, j] = new_values

#Line successive over-relaxation method:
def LSOR(temp_old, temp_new):
    omega = 1.27
    for j in range(1,columns-1):
        a = np.ones(rows-3) * -1.0 * omega
        b = np.ones(rows-2) * 2 * (1.0 + beta**2)
        c = np.ones(rows-3) * -1.0 * omega
        d = np.zeros(rows-2)
        for i in range(1,rows-1):
            d[i-1] = (omega * beta**2 * (temp_new[i][j-1] + temp_old[i][j+1])) + (2*(1 - omega)*(1 + beta**2)*temp_old[i][j])
        d[0] += 30*omega
        d[-1] += 100*omega

        new_values = tdma_solve(a,b,c,d)
        temp_new[1:rows-1, j] = new_values

#ADI Scheme:
def ADI(temp_old, temp_new):
    temp_mid = temp_old.copy()
    #Along X:
    for j in range(1,columns-1):
        a = np.ones(rows-3) * -1.0 
        b = np.ones(rows-2) * 2 * (1.0 + beta**2)
        c = np.ones(rows-3) * -1.0 
        d = np.zeros(rows-2)
        for i in range(1,rows-1):
            d[i-1] = beta**2 * (temp_mid[i][j-1] + temp_old[i][j+1])
        d[0] += 30
        d[-1] += 100
        new_values = tdma_solve(a,b,c,d)
        temp_mid[1:rows-1, j] = new_values
    
    #ALong Y:
    for i in range(1,rows-1):
        a = np.ones(columns-3) * -1.0 * beta**2
        b = np.ones(columns-2) * 2 * (1.0 + beta**2)
        c = np.ones(columns-3) * -1.0 * beta**2
        d = np.zeros(columns-2)
        for j in range(1,columns-1):
            d[j-1] = temp_new[i-1][j] + temp_mid[i+1][j]
        d[0] += 30 * beta**2
        d[-1] += 30 * beta**2
        new_values = tdma_solve(a,b,c,d)
        temp_new[i, 1:columns-1] = new_values

#ADI Scheme with relaxation:
def ADIwRlx(temp_old, temp_new):
    omega = 1.31
    temp_mid = temp_old.copy()
    #Along X:
    for j in range(1,columns-1):
        a = np.ones(rows-3) * -1.0 * omega
        b = np.ones(rows-2) * 2 * (1.0 + beta**2)
        c = np.ones(rows-3) * -1.0 * omega
        d = np.zeros(rows-2)
        for i in range(1,rows-1):
            d[i-1] = omega * beta**2 * (temp_mid[i][j-1] + temp_old[i][j+1]) + 2*(1-omega)*(1+beta**2)*temp_old[i][j]
        d[0] += omega * 30
        d[-1] += omega * 100
        new_values = tdma_solve(a,b,c,d)
        temp_mid[1:rows-1, j] = new_values
    
    #ALong Y:
    for i in range(1,rows-1):
        a = np.ones(columns-3) * -1.0 * beta**2 * omega
        b = np.ones(columns-2) * 2 * (1.0 + beta**2)
        c = np.ones(columns-3) * -1.0 * beta**2 * omega
        d = np.zeros(columns-2)
        for j in range(1,columns-1):
            d[j-1] = omega * (temp_new[i-1][j] + temp_mid[i+1][j]) + 2*(1-omega)*(1+beta**2)*temp_mid[i][j]
        d[0] += 30 * beta**2 * omega
        d[-1] += 30 * beta**2 * omega
        new_values = tdma_solve(a,b,c,d)
        temp_new[i, 1:columns-1] = new_values

# solve_and_plot(Point_Gauss_Seidel)
# solve_and_plot(Line_Gauss_Seidel)
# solve_and_plot(ADI)
# solve_and_plot(PSOR)
# solve_and_plot(LSOR)
solve_and_plot(ADIwRlx)