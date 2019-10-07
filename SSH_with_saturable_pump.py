"""
Copyright (c) 2017, Emiliano Cancellieri  / git: ecancellieri
This module defines the imputs and main code to evaluate
the time evolution of a set of differential equations
correspoding to a SSH chain with saturable pump on first
and last sites, and linear decay on all sites
It compares three different numerical solvers
"""
import numpy as np
import subroutines_ssh
import matplotlib.pyplot as plt

############## INPUTS ############
nsites   = 20           # Set number of (even) sites in the SSH chain
ttot     = 1000         # Simulation time
nt       = 20000        # Number of steps in time
tsteady  = 900          # Time after which a quasi-steady state is reached
rndness  = 0.0          # Degree of randomness/disorder in pump, decay, and hopping
taui     = 0.7          # Hopping probability intra-dimer
taue     = 1.0          # Hopping probability extra-dimer
nla      = 0.115j       # Imaginary nonlinear pump on end a-site
nlb      = 0.0j         # Imaginary nonlinear pump on end b-site
lina     = -0.1         # Linear imaginary part on a-sites (decay)
linb     = -0.1         # Linear imaginary part on b-sites (decay)
popini   = 0.1          # Occupation of each site at t=0
solver   = 'ue'       # Available: fork (Fourth order Runge Kutta)
						# mdp (Mid-point predictor)
						# ue (Unitary Euler)
#################################

# set linear time vector
t = np.linspace(0.0, ttot, nt)
# set random seed for reproducibility
np.random.seed(1234)

# generates the matrices defining the hamiltonian
rndnl, rndlin, rndtaumat = subroutines_ssh.get_hamiltonian_with_disorder(taui, taue, nla, nlb, lina, linb, rndness, nsites)

# generates random initial conditions
z0 = subroutines_ssh.randominitial(popini, nsites, rndness)

# Evaluate time evolution with different solvers
if solver == 'fork':
	# Fourth order Runge Kutta
	z = subroutines_ssh.fork_rnd(z0, rndtaumat, rndnl, rndlin, nsites, t)
elif solver == 'mdp':
	# Mid point predictor
	z = subroutines_ssh.mpp_rnd(z0, rndtaumat, rndnl, rndlin, nsites, t)
elif solver == 'ue':
	# Unitary Euler
	z = subroutines_ssh.uni_euler(z0, rndtaumat, rndnl, rndlin, nsites, t)
else:
	print('Solver not recognized')

# Plot results
plt.figure('Time evolution and spectra',figsize=(9,2))
subroutines_ssh.myplot(z, rndtaumat, rndnl, rndlin, nsites, t, tsteady)
# Save figure
plt.savefig('Time_evolution_and_spectra.pdf',figsize=(9,2))
plt.show()


