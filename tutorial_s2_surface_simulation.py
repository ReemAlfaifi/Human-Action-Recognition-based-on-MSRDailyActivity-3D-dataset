#Import a bunch of stuff to ease command line usage
from tvb.simulator.lab import *
#Initialise a Model, Coupling, and Connectivity.
oscillator = models.Generic2dOscillator()
white_matter = connectivity.Connectivity.from_file('/home/reem/Downloads/TVB_Linux_2.3/TVB_Distribution/tvb_data/lib/python3.7/site-packages/tvb_data/connectivity/Quantum_connectivity_76.zip')
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=numpy.array([0.014]))

#Initialise an Integrator
heunint = integrators.HeunDeterministic(dt=2**-4)

#Initialise a surface
default_cortex = cortex.Cortex.from_file()
default_cortex.coupling_strength = numpy.array([2**-10])
default_cortex.local_connectivity = local_connectivity.LocalConnectivity.from_file()

#Initialise some Monitors with period in physical time
mon_tavg = monitors.TemporalAverage(period=2**-2)
mon_savg = monitors.SpatialAverage(period=2**-2)
# load the default region mapping
rm = region_mapping.RegionMapping.from_file('/home/reem/Downloads/TVB_Linux_2.3/TVB_Distribution/tvb_data/lib/python3.7/site-packages/tvb_data/regionMapping/regionMapping_16k_76.txt')
mon_eeg = monitors.EEG.from_file()
mon_eeg.region_mapping=rm
#Bundle them
what_to_watch = (mon_tavg, mon_savg, mon_eeg)

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and surface.
sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                          coupling = white_matter_coupling, 
                          integrator = heunint, monitors = what_to_watch,
                          surface = default_cortex)

sim.configure()

#Perform the simulation
tavg_data = []
tavg_time = []
savg_data = []
savg_time = []
eeg_data = []
eeg_time = []
for tavg, savg, eeg in sim(simulation_length=2**7):
    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])
    
    if not savg is None:
        savg_time.append(savg[0])
        savg_data.append(savg[1])
    
    if not eeg is None:
        eeg_time.append(eeg[0])
        eeg_data.append(eeg[1])
        
TAVG = numpy.array(tavg_data)
SAVG = numpy.array(savg_data)
EEG = numpy.array(eeg_data)


