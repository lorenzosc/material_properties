#Pacotes
import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.integrate as integrate


#%% Inital information: defining ranges of experimental data

T_i = float(input("Initial Temperature for experimental data (K): "))
T_f = float(input("Final Temperature for experimental data (K): "))
dT = float(input("Temperature increment for experimental data (K): "))
H_i = float(input("Initial Magnetic Field (T): "))
H_f = float(input("Final Magnetic Field (T): "))
dH = float(input("Magnetic Field increment (T): "))

#%% Step 1: importing experimental data

Nline = int(1 + (T_f-T_i)/dT) #Number of lines - represents experimental temepratures
Ncol = int(1 + (H_f - H_i)/dH) #Number of columns - represents experimental fields
#vectors to read original data
T_exp = np.zeros([Nline]) 
Tad_exp = np.zeros([Nline])
H_exp = np.linspace(H_i, H_f, 1+int(np.round((H_f - H_i)/dH))) 
s0_exp = np.zeros([Ncol]) 
c_exp = np.zeros([Nline, Ncol])
dS_exp = np.zeros([Nline, Ncol])
dTad_exp = np.zeros([Nline, Ncol])

#reading for temperature and experimental specific heat
path = str(input("Directory and name of .txt file with experimental Temperature and specific heat data: "))
j=0
with open(path, "r") as ref_file:
    for line in ref_file:  
        values = line.split()
        if j > 0: #do not read the header
            T_exp[j-1] = values[0]
            c_exp[j-1,:] = values[1:]
        j += 1    

#reading for dS x T x H experimental data
path = str(input("Directory and name of .txt file with dS x T x H experimental data: "))
j=0
with open(path, "r") as ref_file:
    for line in ref_file:  
        values = line.split()
        if j > 0: #do not read the header
            dS_exp[j-1,1:] = values[1:]
        j += 1

#reading for dTad x T x H experimental data
path = str(input("Directory and name of .txt file with dTad x T x H experimental data: "))
j=0
with open(path, "r") as ref_file:
    for line in ref_file:  
        values = line.split()
        if j > 0: #do not read the header
            Tad_exp[j-1] = values[0]
            dTad_exp[j-1,1:] = values[1:]
        j += 1



#%% Option: processing experimental data - if necessary

#Do you wish to reduce the available experimental data? 

reduce_data = str(input("Do you wish to reduce the available experimental data? (y/n)\n"))
#If not, keep the values of Trange and Tinc equals to zero (0)
if reduce_data == "n":
    Trange = 0
    Tinc = 0

Tcut_lower = 0
Tcut_upper = 0 
# Trange = reduce temperature range
if reduce_data == "y":
    reduce_range = str(input("Do you wish to reduce the temperature range? (y/n)\n"))
    Trange = 0
    if reduce_range == "y":
        Trange = 1
        Tcut_lower = int(input("Set the temperature interval to cut from T_i towards T_f: "))
        Tcut_upper = int(input("Set the temperature interval to cut from T_f towards T_i: "))
    increase_increment = str(input("Do you wish to increase the temperature increment? (y/n) "))
    Tinc = 0
    if increase_increment == "y":
        Tinc = 1
        dTnew = float(input("Set new temperature increment: "))

#changing Temperature range
Cut_lower = Tcut_lower/dT #number of lines to cut from T_i toward T_f
Cut_upper = Tcut_upper/dT #number of lines to cut from T_f toward T_i
Nline_new = int(Nline - Cut_lower - Cut_upper)
     
#changing dT increment
if Tinc == 1:
    Nline_new = int(1 + ((T_f-Tcut_upper)-(T_i+Tcut_lower))/dTnew)
    jump = int(dTnew/dT)
    
#vectors to processed experimental data 
T_expp = np.zeros([Nline_new]) 
c_expp = np.zeros([Nline_new, Ncol])

#Using original data
if Trange == 0 and Tinc == 0:
    T_expp = T_exp.copy()
    c_expp = c_exp.copy()

#Using an alternative temperature range
if Trange == 1 and Tinc == 0:
    for i in range(Nline_new):
        ii = int(i+Cut_lower)
        T_expp[i] = T_exp[ii] 
        for j in range(Ncol):
            c_expp[i,j] = c_exp[ii,j]
    
#Using an alternative temperature increment
if Trange == 0 and Tinc == 1:
    cont = 0 
    for i in range(Nline_new):
        if i == 0:
            ii = i
        else:
            ii = int(cont*jump)
        T_expp[i] = T_exp[ii] 
        for j in range(Ncol):
            c_expp[i,j] = c_exp[ii,j]
        cont += 1

#Using an alternative temperature increment and an alternative temperature range
if Trange == 1 and Tinc == 1:
    cont = 0 
    for i in range(Nline_new):
        if i == 0:
            ii = int(i+Cut_lower)
        else:
            ii = int(cont*jump+Cut_lower)
        T_expp[i] = T_exp[ii] 
        for j in range(Ncol):
            c_expp[i,j] = c_exp[ii,j]
        cont += 1  
        
        
#determining the initial value of entropy s0 x H @ T = Tinitial
s0_exp[0] = c_expp[0,0]/2
for i in range(1, Ncol):
    s0_exp[i] = s0_exp[0] + dS_exp[0,i]
    
#%% - plot c x T x H from original data with modifications or not

#preparating color generator for plots
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

plot_original_data = str(input("Do you wish to plot the loaded data? (y/n)\n"))
if plot_original_data == "y":
    plt.figure(figsize=(12, 8))
    cmap = get_cmap(len(c_expp[0])+1)
    for i in range(len(c_expp[0])):
        plt.scatter(T_expp[:], c_expp[:,i], label='curve', color=cmap(i))
    plt.xlabel('T (K)')
    plt.ylabel('c (J/kgK)')
    plt.show()

#%% Step 2: fitting mathematical function for specific heat

#Transformation in C data
peakC = np.amax(c_expp[:])
maxT = np.amax(T_expp[:])+20


c_transform = np.zeros([Nline_new,Ncol]) 
for j in range(1,Ncol):
    for i in range(Nline_new):
        c_transform[i,j] = (c_expp[i,j]/peakC)**2* (maxT - T_expp[i])

#defining Poly Fit
#here x stands for temperature
def PolyFitC(x, kk0, kk1, kk2, kk3, kk4, kk5, kk6):
    return  (kk0+kk2*x**(0.5)+kk4*x+kk6*x**(1.5))/(1+kk1*x**(0.5)+kk3*x+kk5*x**(1.5))

#defining the vectors to store the fitting curves constants to each experimental field
k0 = np.zeros([Ncol])  
k1 = np.zeros([Ncol])  
k2 = np.zeros([Ncol])  
k3 = np.zeros([Ncol])  
k4 = np.zeros([Ncol])  
k5 = np.zeros([Ncol])  
k6 = np.zeros([Ncol]) 
#defining vector for the fitted specific heat
c_polyfitT = np.zeros([Nline_new,Ncol]) 
c_polyfit = np.zeros([Nline_new,Ncol]) 

#sweeping in magnetic fields
for j in range(1,Ncol):
    #fitting parameters
    par, par_cov = optimize.curve_fit(PolyFitC, T_expp[:], c_transform[:,j], maxfev=10000)
    
    #atributing values to parameters fitted
    k0[j] = par[0]
    k1[j] = par[1]
    k2[j] = par[2]
    k3[j] = par[3]
    k4[j] = par[4]
    k5[j] = par[5]
    k6[j] = par[6]
 
    #sweeping in temperature
    for i in range(Nline_new):
        x  = T_expp[i]
        c_polyfitT[i,j] = (k0[j]+k2[j]*x**(0.5)+k4[j]*x+k6[j]*x**(1.5))/(1+k1[j]*x**(0.5)+k3[j]*x+k5[j]*x**(1.5))
        c_polyfit[i,j] =  peakC*(c_polyfitT[i,j]/(maxT - T_expp[i]))**0.5

#%% Deleting c range after fitting
Tcut_upper = int(input("After the fitting, how many degrees from the higher temperature side do you wish to cut from the analysis?\n"))
Cut_upper = Tcut_upper/dT #number of lines to cut from T_f toward T_i
Nline_newC = int(np.round(Nline_new - Cut_upper)) 
Nline_new = Nline_newC   

#vectors to process experimental data 
T_expC = np.zeros([Nline_newC]) 
c_polyfitC = np.zeros([Nline_newC, Ncol])

#T_expC = T_exp[:Nline_new]
#c_polyfitC = c_polyfit[:Nline_new,:]

for i in range(Nline_new):
    T_expC[i] = T_exp[i] 
    for j in range(Ncol):
        c_polyfitC[i,j] = c_polyfit[i,j]


#%% - plot c x T x H - comparison between fitted and experimental

comparison_original_fitted = str(input("Do you wish to plot the comparison between fitted and experimental data? (y/n)\n"))
if comparison_original_fitted == "y":
    plt.figure(figsize=(12, 8))
    cmap = get_cmap(len(c_expp[0])+2)
    for i in range(len(c_expp[0])):
        plt.scatter(T_expp[:], c_expp[:,i], label='curve', color=cmap(i))
    plt.plot(T_expC[:], c_polyfitC[:,1:], label='curve', color=cmap(i+1))
    plt.xlabel('T (K)')
    plt.ylabel('c (J/kgK)')
    plt.show()
#%% - Step 3: Interpolation of k´s and calculating c x T for intermediated fields

#Select interpolation type
SelectType = int(input("Set type of interpolation.\n1 for linear, 2 for quadratic, 3 for cubic\n"))
switch = {1: 'linear', 2: 'quadratic', 3: 'cubic'}
functype = switch[SelectType]

#Defining the desired fields and temperatures increments
dH_new = float(input("Insert field increment for interpolation of the intermediate fields\n"))
Ncol_new = int(1 + (H_f-H_i)/dH_new)
Ncol_lowH = int(1 + (H_exp[1]-H_i)/dH_new)
dT_new = float(input("Insert temperature increment for interpolation of the intermediate temperature\n"))
Nline_new = int(1 + ((T_f-Tcut_upper)-(T_i + Tcut_lower))/dT_new)

#Defining new vector and matrices for T, H and c
T_fit = np.zeros([Nline_new]) 
H_fit = np.zeros([Ncol_new]) 
c_fit = np.zeros([Nline_new, Ncol_new])
c_fitT = np.zeros([Nline_new, Ncol_new])
c_lowH = np.zeros([Ncol]) 
H_lowH = np.zeros([Ncol_lowH])

#For low fields
for i in range(Nline_new):
    for j in range(Ncol):
        c_lowH[j] = c_expp[i,j]
    
    for jj in range(Ncol_lowH):
        H_lowH[jj] = jj*dH_new
        y = jj*dH_new
        interp_lowH = interp1d (H_exp, c_lowH, kind=functype) 
        c_fit[i,jj] = interp_lowH(y) 

#For high fields
for i in range(Nline_new):
    T_fit[i] = (T_i + Tcut_lower) + i*dT_new

for j in range(Ncol_new):
    H_fit[j] = j*dH_new

#interpolation of the k´s coefficients
for i in range(Nline_new):
    for j in range(Ncol_lowH-1,Ncol_new):
        x  = T_fit[i]
        y = H_fit[j]
            
        interp_k0 = interp1d (H_exp, k0, kind=functype) 
        kk0 = interp_k0(y) 
        interp_k1 = interp1d (H_exp, k1, kind=functype) 
        kk1 = interp_k1(y) 
        interp_k2 = interp1d (H_exp, k2, kind=functype) 
        kk2 = interp_k2(y) 
        interp_k3 = interp1d (H_exp, k3, kind=functype) 
        kk3 = interp_k3(y) 
        interp_k4 = interp1d (H_exp, k4, kind=functype) 
        kk4 = interp_k4(y) 
        interp_k5 = interp1d (H_exp, k5, kind=functype) 
        kk5 = interp_k5(y) 
        interp_k6 = interp1d (H_exp, k6, kind=functype) 
        kk6 = interp_k6(y) 
            
        c_fitT[i,j] = (kk0+kk2*x**(0.5)+kk4*x+kk6*x**(1.5))/(1+kk1*x**(0.5)+kk3*x+kk5*x**(1.5))
        c_fit[i,j] =  peakC*(c_fitT[i,j]/(maxT - T_expp[i]))**0.5

#%% - plot c x T x H - fitted curves with intermediate fiels

plot_intermediate_fields = str(input("Do you wish to plot the interpolated magnetic fields? (y/n)\n"))
if plot_intermediate_fields == "y":
    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], c_fit[:,:], label='curve', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('c (J/kgK)')
    plt.show()

#%% - Step 4: Obtaining T-s diagram for all the considered fields 

#defining polinomial function (third degree) to compute s0 based on experimental data
def Poly2(x, cc0, cc1, cc2, cc3):
    return cc0 + cc1*x + cc2*x**2 + cc3*x**3

#fitting parameters
par, par_cov = optimize.curve_fit(Poly2, H_exp, s0_exp, maxfev=7000)

#Defining the new vector s0
s0 = np.zeros([Ncol_new]) 

#obtaining values for s0 to any considered field in J/kgK
for j in range(Ncol_new):
    x = H_fit[j]
    s0[j] = par[0] + par[1]*x + par[2]*x**2 + par[3]*x**3

#Defining new matrices and vectors
cT_aux = np.zeros([Nline_new]) #auxiliary parameter c/T
s = np.zeros([Nline_new, Ncol_new]) #total entropy in J/kgK

for j in range(Ncol_new):
    for i in range(Nline_new):
        cT_aux[i] =  c_fit[i,j]/T_fit[i]
        s[i,j] = s0[j] + integrate.simps(cT_aux, x=T_fit, dx=dT_new,even='avg')
        #It could be necessary apply the following filters
        #@ very low fields or in regions where dS tends to zero, some 
        #inconsistencies can be verified in T-s diagram
        if s[i,j] > s[i,0]:
            s[i,j] = s[i,0]
        if s[i,j] < s[i-1,j]:
            s[i,j] = s[i-1,j]
        

#%% - plot s x T x H - or T-s diagrama - to all considered fields

plot_T_S = str(input("Do you wish to plot the s x T x H diagram? (y/n)\n"))
if plot_T_S == "y":
    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], s[:,:], label='curve', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('s (J/kgK)')
    plt.show()
#%% Step 5: determining dS x T x H, and compare with experimental data 

#defining matrix dS (delta S) in J/kgK
ds = np.zeros([Nline_new, Ncol_new])

for j in range(Ncol_new):
    for i in range(Nline_new):
        ds[i,j] = s[i,j] - s[i,0]

#%% - plot ds x T x H to all considered fields
plot_ds = str(input("Do you wish to plot the ds x T x H diagram? (y/n)\n"))
if plot_ds == "y":
    plt.figure(figsize=(12, 8))
    cmap = get_cmap(len(dS_exp[0])+1)
    for i in range(1, len(dS_exp[0])):
        plt.scatter(T_exp[:], dS_exp[:,i], label='exp', color=cmap(i))
    col = 1
    for j in range(Ncol_new):
        if H_fit[j] == H_exp[col]:
            col += 1
            plt.plot(T_fit[:], ds[:,j], label='fit', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('ds (J/kgK)')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], ds[:,:], label='curve', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('ds (J/kgK)')
    plt.show()
#%% Step 5 continuation: determining dTad x T x H, and compare with experimental data 

#defining matrix dTad in K
dTad = np.zeros([Nline_new, Ncol_new])

#finding dTad from T-s diagram
for j in range(1,Ncol_new):
    for i in range(Nline_new):
        #initial
        T_ini = T_fit[i]
        s_ini = s[i,0]
        #final
        s_fin = s_ini
        if s_fin > s[Nline_new-1,j]:
            T_fin = T_ini
        else:
            interp_Tf = interp1d (s[:,j], T_fit[:], kind='cubic') 
            T_fin = interp_Tf(s_fin)
        
        dTad[i,j] = T_fin - T_ini
        
#finding dTad from T-s diagram - removing field
dTad_rev = np.zeros([Nline_new, Ncol_new])
for j in range(1,Ncol_new):
    for i in range(Nline_new):
        #initial
        T_ini = T_fit[i]
        s_ini = s[i,j]
        #final
        s_fin = s_ini
        if s_fin < s[0,0]:
            T_fin = T_ini
        else:
            interp_Tf = interp1d (s[:,0], T_fit[:], kind='cubic') 
            T_fin = interp_Tf(s_fin)
        
        dTad_rev[i,j] = -(T_fin - T_ini)
    
#%% - plot dTad x T x H to all considered fields

plot_dTad = str(input("Do you wish to plot the adiabatic temperature difference? (y/n)\n"))
if plot_dTad == "y":    
    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], dTad[:,:], label='curve', color='black')
    plt.plot(T_fit[:], dTad_rev[:,:], label='curve', color='orange')
    plt.xlabel('T (K)')
    plt.ylabel('dTad (K) - from T-s diagram')
    plt.show()            
    
    plt.figure(figsize=(12, 8))
    cmap = get_cmap(len(dTad_exp[0])+1)
    for i in range(1, len(dTad_exp[0])):
        plt.scatter(Tad_exp[:], dTad_exp[:,i], label='exp', color=cmap(i))
    plt.xlim(T_i, T_f)
    col = 1
    for j in range(Ncol_new):
        if H_fit[j] == H_exp[col]:
            col += 1
            plt.plot(T_fit[:], dTad[:,j], label='fit', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('dTad (K) - from T-s diagram')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for i in range(1, len(dTad_exp[0])):
        plt.scatter(Tad_exp[:], dTad_exp[:,i], label='exp', color=cmap(i))
    plt.xlim(T_i, T_f)
    col = 1
    for j in range(Ncol_new):
        if H_fit[j] == H_exp[col]:
            col += 1
            plt.plot(T_fit[:], dTad_rev[:,j], label='fit', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('dTad (K) - from T-s diagram')
    plt.show()

#%% Step 6: determining M x T x H from T-s diagram ONLY for the experimental (reference) fields

#The first step is to upload experimental data for Magnetization evaluated at the experimental
#fields, but at the highest temperature of interest 
#This means, if you did not cut the upper region is at T = T_f; if you did cut the upper region you need to correct the T_f

#reading for M0 x H experimental data
path = str(input("Directory and name of .txt file with M0 x H experimental data: "))
j=0
with open(path, "r") as ref_file:
    for Tlines, line in enumerate(ref_file):
        pass
    M0 = np.zeros([Tlines, Ncol+1])
    MM0 = np.zeros([Ncol])
    for line in ref_file:  
        values = line.split()
        if j > 0: #do not read the header
            M0[j-1, 0] = values[0]
            M0[j-1, 1] = values[1]
            M0[j-1, 2] = values[2]
            M0[j-1, 3] = values[3]
            M0[j-1, 4] = values[4]
            M0[j-1, 5] = values[5]
        j += 1

#picking the correct M0 values due to possible temperature range correction
for i in range(Tlines):
    if M0[i, 0] == (T_f-Tcut_upper):
        for col in range(Ncol):
            MM0[col] = M0[i, col+1]

#defining the new matrix s_fields, which only accounts the experimental fields values
s_fields = np.zeros([Nline_new, Ncol])

#Calculating ds/dT and ds/dH from T-s diagram ONLY for the experimental fields
col = 0
for j in range(Ncol_new):
    if H_fit[j] == H_exp[col]:
        for i in range(Nline_new):
            s_fields[i,col] = s[i,j]     
        col += 1

dsdT, dsdH = np.gradient(s_fields)

#defining matrices and vectors
dM = np.zeros([Nline_new]) #vector delta M
M = np.zeros([Nline_new, Ncol]) #matrix magnetization
dsdHarray = np.zeros([Nline_new]) #auxiliary paramenter dsdH

#Calculating the Magnetization values for the experimental fields and in a range of temperatures
#Note that the sweeping on temperature starts from the higher temperature towards to the lower value (T_i)
#This is in a try to have less impacts related to demagnetizing fields
for j in range(Ncol):
    for i in range(Nline_new-1,-1,-1):
        dM[i] = 0
        dsdHarray[i] = 0
    for i in range(Nline_new-1,-1,-1):
        dsdHarray[i] = dsdH[i,j]/dH
        if dsdHarray[i] > 0:
              dsdHarray[i] = - dsdHarray[i]
        dM[i] =  integrate.simps(dsdHarray, x=T_fit, dx=dT_new,even='avg')
        M[i,j] = MM0[j] - dM[i] 


#%% plot M x T curves for experimental fields

plot_M_T = str(input("Do you wish to plot magnetization curves for experimental fields? (y/n)\n"))
if plot_M_T == "y":
    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], M[:,:], label='curve', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('M (emu/g)')
    plt.show()  
#%% Step 7: fitting for the MxT curve using the experimental fields
#The MxT curve has a complex behaviour being not trivial to find a good fitting curve to use
#thus, it is necessary to perform some transformations to the initial MxT curves
#i) to transform M in a non-dimensionlized variable
#ii) invert the curve trend in function to temperature

#i) non-dimensionalization and ii) inverting the curve trend
#creating matrix Mad and Minv
Mad = np.zeros([Nline_new, Ncol]) #dimesionless M
Minv = np.zeros([Nline_new, Ncol]) #inverted dimensionless M
Mmax = np.zeros([Ncol]) #vector to store Mmax values
Mmin = np.zeros([Ncol]) #vector to store Mmin values



#next, fitting parameters for Min, Mmax and Minv curve can be found 
#fitting an equation for Minv - Gaussian Cumulative
#x stands for Temperature
def Sigmoid(x, aa0, aa1, aa2, aa3):
     return aa0 + aa1/(1+np.exp(-(x-aa2)/aa3)) 
 

a0 = np.zeros([Ncol])  
a1 = np.zeros([Ncol])  
a2 = np.zeros([Ncol])  
a3 = np.zeros([Ncol]) 

for j in range(Ncol):
    par, par_cov = optimize.curve_fit(Sigmoid, T_fit[:], M[:,j], p0=[-0.394,66.1,289.78,-6.516], maxfev=7000)
    
    a0[j] = par[0]
    a1[j] = par[1]
    a2[j] = par[2]
    a3[j] = par[3]


#%% Step 8: Interpolation of a´s and calculating M x T for intermediated fields

#Select interpolation type
SelectType = int(input("Set type of interpolation for the parameters of the fitting M curves.\n1 for linear, 2 for quadratic, 3 for cubic\n"))
switch = {1: 'linear', 2: 'quadratic', 3: 'cubic'}
functype = switch[SelectType]

#creating the new matrices   
eq = np.zeros([Nline_new, Ncol_new])
M_fit = np.zeros([Nline_new, Ncol_new])


for i in range(Nline_new):
    for j in range(Ncol_new):
        x  = T_fit[i]
        y = H_fit[j]
            
        interp_a0 = interp1d (H_exp, a0, kind=functype) 
        aa0 = interp_a0(y) 
        interp_a1 = interp1d (H_exp, a1, kind=functype) 
        aa1 = interp_a1(y) 
        interp_a2 = interp1d (H_exp, a2, kind=functype) 
        aa2 = interp_a2(y) 
        interp_a3 = interp1d (H_exp, a3, kind=functype) 
        aa3 = interp_a3(y) 
            
        interp_Mmax = interp1d (H_exp, Mmax, kind=functype) 
        MMmax = interp_Mmax(y) 
        interp_Mmin = interp1d (H_exp, Mmin, kind=functype) 
        MMmin = interp_Mmin(y) 
            
        #Returing to Magnetization values for all fields - the following equation stands to the inversion fuction for sech
        M_fit[i,j] = aa0 + aa1/(1+np.exp(-(x-aa2)/aa3)) 
            
        #Filter: this may be needed for 0 T magnetization
        if M_fit[i,j] < 0:
            M_fit[i,j] = 0



#%% plot M x T curves to all fields

plot_M_T_all = str(input("Do you wish to plot magnetization curve for all fields? (y/n)\n"))
if plot_M_T_all == "y":
    plt.figure(figsize=(12, 8))
    plt.plot(T_fit[:], M_fit[:,:], label='curve', color='black')
    plt.xlabel('T (K)')
    plt.ylabel('Minv (-)')
    plt.show() 

#%% Step 9: calculating dS curves from M x T and verification with interpolated and experimental data

#Caclulating the quantities dM/dT and dM/dH
#for experimental fields
dMdTexp, dMdHexp = np.gradient(M)
#to all fields
dMdT, dMdH = np.gradient(M_fit)

#creating matrices
dS_MxTexp = np.zeros([Nline_new, Ncol])
dS_MxT = np.zeros([Nline_new, Ncol_new])

#dS for experimental fields
for i in range(Nline_new): 
    Sum = 0
    for j in range(Ncol-1):
        Sum = Sum + (1/2)*(dMdTexp[i,j] + dMdTexp[i,j+1])*dH/dT_new
        dS_MxTexp[i,j] = Sum
        
#dS for all fields
for i in range(Nline_new): 
    Sum = 0
    for j in range(Ncol_new-1):
        Sum = Sum + (1/2)*(dMdT[i,j] + dMdT[i,j+1])*dH_new/dT_new
        dS_MxT[i,j] = Sum

#%% plot for verification
print("Final plots for verification")
plt.figure(figsize=(12, 8))
cmap = get_cmap(len(dS_exp)+2)
for i in range(1, len(dS_exp)):
    plt.scatter(T_exp[:], dS_exp[:,i], label='exp', color=cmap(i))
plt.plot(T_fit[:], dS_MxTexp[:,:], label='fit', color=cmap(i+1))
plt.xlabel('T (K)')
plt.ylabel('ds (J/kgK)')
plt.show()

#%% Step 10 - exporting data
functions.output_data(Nline_new, Ncol_new, T_fit, c_fit, s, M_fit)
