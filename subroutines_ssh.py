"""
Copyright (c) 2017, Emiliano Cancellieri / git: ecancellieri

This module defines the subroutines to evaluate
the time evolution of a set of differential equations
correspoding to a SSH chain with saturable pump on first
and last sites, and linear decay on all sites
It compares three different numerical solvers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift

########################################################################
# Function defining the random gain, losses, particle-particle interaction
# and tau probability for a SSH chain with a "defece" at the centre 
########################################################################
def get_hamiltonian_with_disorder( taui, taue, nla, nlb, lina, linb, rndness, nsites):
	# Define pump and decay terms
	rndnla = np.zeros(nsites, dtype=complex)
	rndnlb = np.zeros(nsites, dtype=complex)
	rndlina = np.zeros(nsites, dtype=complex)
	rndlinb = np.zeros(nsites, dtype=complex)
	for i in range(nsites):
		if i % 2 == 0:
			rndnla[i] = np.real(nla)*(1+(np.random.random()-0.5)*rndness)+\
						np.imag(nla)*1j*(1+(np.random.random()-0.5)*rndness)
			rndlina[i] = lina*(1+(np.random.random()-0.5)*rndness)
		else:
			rndnlb[i] = np.real(nlb)*(1+(np.random.random()-0.5)*rndness)+\
						np.imag(nlb)*1j*(1+(np.random.random()-0.5)*rndness)
			rndlinb[i] = linb*(1+(np.random.random()-0.5)*rndness)
	rndnl = rndnla + rndnlb
	rndlin = rndlina + rndlinb

	# Pump only on first and last
	if i != 0 or i != nsites-1:
		rndnl[i]=0

	# Define hopping rate between sites
	upper1 = np.zeros(nsites-1)
	lower1 = np.zeros(nsites-1)
	for i in range( nsites-1 ):
		if i % 2 == 0:
			upper1[i] = taui*(1+(np.random.random()-0.5)*rndness)
			lower1[i] = taui*(1+(np.random.random()-0.5)*rndness)
		else:
			upper1[i] = taue*(1+(np.random.random()-0.5)*rndness)
			lower1[i] = taue*(1+(np.random.random()-0.5)*rndness)
	rndtaumat=np.diag(upper1,1)+np.diag(lower1,-1)
	return rndnl, rndlin, rndtaumat

#################################################################################
# Function defining the random initial conditions for the SSH chain with a defect 
#################################################################################
def randominitial(popini, nsites, rndness):
	# Complex random initial conditions
	rndness = 1.0
	z0 = np.array([popini*(1+(np.random.random()-0.5)*rndness)*np.exp(1j*2.0*\
		 np.pi*(1+(np.random.random()-0.5)*rndness)) for i in range(nsites)],dtype=complex )

	return z0

#######################################################################################
# Function defining the Jacobian for the set of differential equations of the SSH chain 
#######################################################################################
def zjac(z, t, rndtaumat, rndnl, rndlin, nsites):
	jac = -1j*rndtaumat + np.diag((rndlin-1j*(1j*np.imag(rndnl))/(1+abs(z)**2)\
		  -1j*np.real(rndnl)*abs(z)**2))
	return jac

##########################
# Fourth order Runge Kutta
##########################
def fork_rnd(z0, rndtaumat, rndnl, rndlin, nsites, t):
	nt = len(t)
	zout = np.zeros([nt,len(z0)],dtype=complex)
	zout[0] = z0
	dt = t[nt-1]/nt
	for i in range(1,nt):
		k1 = np.dot(zjac(zout[i-1], 0, rndtaumat, rndnl, rndlin, nsites),zout[i-1])
		k2 = np.dot(zjac(zout[i-1]+0.5*k1*dt, 0, rndtaumat, rndnl, rndlin, nsites),(zout[i-1]+0.5*k1*dt))
		k3 = np.dot(zjac(zout[i-1]+0.5*k2*dt, 0, rndtaumat, rndnl, rndlin, nsites),(zout[i-1]+0.5*k2*dt))
		k4 = np.dot(zjac(zout[i-1]+1.0*k3*dt, 0, rndtaumat, rndnl, rndlin, nsites),(zout[i-1]+1.0*k3*dt))
		zout[i] = zout[i-1] + (k1+2*k2+2*k3+k4)*dt/6
	return zout

####################
# Midpoint predicotr
####################
def mpp_rnd(z0, rndtaumat, rndnl, rndlin, nsites, t):
	nt = len(t)
	zout = np.zeros([nt,len(z0)],dtype=complex)
	zout[0] = z0
	dt = t[nt-1]/nt
	for i in range(1,nt):
		k1 = np.dot(zjac(zout[i-1], 0, rndtaumat, rndnl, rndlin, nsites),zout[i-1])
		k2 = np.dot(zjac(zout[i-1]+0.5*k1*dt, 0, rndtaumat, rndnl, rndlin, nsites),(zout[i-1]+0.5*k1*dt))
		zout[i] = zout[i-1] + k2*dt
	return zout

##############
# Untary Euler
##############
def uni_euler(z0, rndtaumat, rndnl, rndlin, nsites, t):
	nt = len(t)
	zout = np.zeros([nt,len(z0)],dtype=complex)
	zout[0] = z0
	dt = t[nt-1]/nt
	I = np.identity(nsites)
	for i in range(1,nt):
		DU = zjac(zout[i-1], 0, rndtaumat, rndnl, rndlin, nsites)*(dt/2)
		U = np.dot(np.linalg.inv(I-DU),I+DU)
		zout[i] = np.dot(U,zout[i-1])
	return zout

###########################################
# Function calculating and ordering the FFT 
###########################################
def myfft(zsteady):
	nsites = np.shape(zsteady)[1]
	ntsteady = np.shape(zsteady)[0]
	afft = np.zeros( ntsteady )
	bfft = np.zeros( ntsteady )
	for i in range( nsites ):
		auxfft = np.absolute(fft(zsteady[:,i]))**2
		if i % 2 == 0:
			afft = afft + auxfft
		else:
			bfft = bfft + auxfft
	afft = fftshift(afft)/ntsteady
	bfft = fftshift(bfft)/ntsteady
	return afft, bfft

#########################
# Function to plot graphs 
#########################
def myplot(z, rndtaumat, rndnl, rndlin, nsites, t, tsteady):
	nt = len(t)
	ntsteady = np.int(nt*(t[-1:]-tsteady)/t[-1:])
	zsteady = z[-ntsteady:]
	# Extract data for time-trace plot:
	# populations on a sites, b sites, all sites
	# Once the steady state is reached
	zaplot = np.zeros(ntsteady)
	zbplot  = np.zeros(ntsteady)
	for j in range(nsites):
		if j % 2 == 0:
			zaplot = zaplot+abs(zsteady[:,j])**2
		else:
			zbplot = zbplot+abs(zsteady[:,j])**2
	ztotplot = zaplot+zbplot
	# Plot time evolution
	ax = plt.subplot2grid((1,6),(0,0), colspan=3)
	tplot = np.linspace(0, t[-1:]-tsteady, ntsteady)
	plt.plot(tplot, np.real(ztotplot), color='black', linewidth=2.0)
	plt.plot(tplot, np.real(zbplot), color='red', linewidth=2.0)
	plt.plot(tplot, np.real(zaplot), color='blue', linewidth=2.0)
	plt.xlabel('time')
	plt.ylabel('$I_{A,B}$')
	plt.ylim(0.0,max(ztotplot)+1)
	ax.get_yaxis().set_tick_params(which='both', direction='in')
	ax.get_xaxis().set_tick_params(which='both', direction='in')
	# Calculate Fourier transform
	fftaplot, fftbplot = myfft(zsteady)
	# define the energy array for the Fourier transform plot
	energy = fftfreq(ntsteady, (t[-1:]-tsteady)/ntsteady)
	energy = fftshift(energy)*2*np.pi
	# Plot Fourier transform
	ax = plt.subplot2grid((1,6),(0,3), colspan=3)
	plt.plot(-energy, fftaplot, color='red', linewidth=2.0)
	plt.plot(-energy, fftbplot, color='blue', linewidth=2.0)
	plt.xlabel('energy')
	plt.ylabel('$I_{A,B}$')
	maxa=max(fftaplot)
	maxb=max(fftbplot)
	plt.xlim(-2*np.pi,2*np.pi)
	plt.ylim(0.001,max(maxa,maxb)+1)
	plt.yticks([])
	ax.get_yaxis().set_tick_params(which='both', direction='in')
	ax.get_xaxis().set_tick_params(which='both', direction='in')
	plt.tight_layout()
	return
