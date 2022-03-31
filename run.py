from rec import REC
from economics import NPV
import postprocess as pp
import time
import os
import json

"""
MESSpy - Run
"""

study_case = 'study case' # str name for results file.pkl
reference_case = 'reference case' # str name for results file.pkl

"""
Input files
"""

path = r'./inputs'

file = 'structure.json'
filepath = os.path.join(path,file)
with open(filepath,'r') as f:
    structure = json.load(f)

file = 'general.json'
filepath = os.path.join(path,file)
with open(filepath,'r') as f:
    general = json.load(f)

time1 = time.time()
 
print('Creating structure')
# Creating initial structure
rec = REC(structure,general) # create REC structure
#rec.reset() # reset REC energy balances

time2 = time.time()
print('Time to create structure {:.2f} seconds'.format(time2-time1))

#%% ###########################################################################
print('Running the model')
time2 = time.time()

# Running the model
rec.REC_energy_simulation() # simulate REC structure
rec.save(study_case) # save results in 'study_case.pkl'
pp.total_balances(study_case)
pp.SOC_plot(study_case)

time3 = time.time()
print('Time to run the model {:.2f} seconds'.format(time3-time2))
  

#%% ###########################################################################
print('Economic analysis') 
time3 = time.time()

file = 'economics.json'
filepath = os.path.join(path,file)
with open(filepath,'r') as f:
    economic_data = json.load(f)

file = 'refcase.json'
filepath = os.path.join(path,file)
with open(filepath,'r') as f:
    structure0 = json.load(f)
    
# Reference case simulation (run only if changed)
rec0 = REC(structure0,general) # create REC
rec0.REC_energy_simulation() # simulate REC 
rec0.save(reference_case) # save results in 'reference_case.pkl'

# Actual economic analysis (It has no sense if simulation_years = 1)
NPV(structure,structure0,study_case,reference_case,economic_data,general['simulation years']) 
pp.NPV_plot()

time4 = time.time()  
print('Time for economic analysis {:.2f} seconds'.format(time4-time3))

