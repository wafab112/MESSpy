import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import math
from numpy import log as ln
from sklearn.linear_model import LinearRegression 
import pandas as pd
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir)))   # temporarily adding constants module path 
from core import constants as c
import scipy.fft
import scipy.optimize

from enum import Enum
import abc

from fuel_cell import FuelCell

class SimpleFuelCell(FuelCell):
    def __init__(self, parameters: FuelCellParameters, timestep_number: int, electrical_efficiency: float, thermal_efficiency:float, timestep = False):
        """Create a simple fuel cell object.
        Absorbs hydrogen and produces electricity, heat and water.
        Calculations are based on nominal power and efficiencies only.
        See self#use

        Parameters
        ----------
        parameters: FuelCellParameters
            All the parameters the fuel cell needs to be instantiated

        electrical_efficiency: float
            A float in [0;1]

        thermal_efficiency: float
            A float in [0;1]

        timestep_number: int
            The number of timesteps the fuel cell should go through

        timestep: boolean = False
            TODO
        """
        super().__init__(parameters, timestep_number, timestep)

        self.h2p_el_eff_in = electrical_efficiency
        self.h2p_th_eff_in = thermal_efficiency

        total_efficiency = self.h2p_el_eff_in + self.h2p_th_eff_in
        if total_efficiency > 1:
            raise ValueError(f"Warning: Total efficiency results to be {total_efficiency*100:.1f} %. Decrease 'electric efficiency' or 'thermal efficiency' so that their sum does not exceed one.")
        print("\nWarning: fuel cell simple model has only a specific consumption value, it does not consider ageing and minimum load")
        self.h2p_eff = (1/(self.h2p_el_eff_in*c.HHVH2*1000))*3600 # [kg/kWh] kg/h of hydrogen produced per kWh of input power            
        self.max_h2_stack   = self.MaxPowerStack/(self.h2p_el_eff_in*c.HHVH2*1000)    # [kg/s] maximum amount of exploitable hydrogen for the considered stack

        print(f"\nThe fuel cell electric efficiency is set equal to {self.h2p_el_eff_in*100:.1f}%, which is equivalent to {round(self.h2p_eff, 3)} kg/kWh (using H2 HHV). "
                  f"The fuel cell thermal efficiency is set equal to {self.h2p_th_eff_in*100:.1f}%. The output heat temperature is not available for the 'simple' model. "
                  f"Thus, the total efficiency is equal to {total_efficiency*100:.1f}%. "
                  f"A fuel cell nominal power of {self.MaxPowerStack:.2f} kW needs to be fed with {self.max_h2_stack*3600:.2f} kg/h of hydrogen.")

    def plot_polarizationpts(self):
        print("Pol. curve not available for simple cell")

    def use(self, step: int, required_power: float, available_hyd: float):
        super().use(step, required_power, available_hyd)

        available_hydrogen = self.available_hydrogen
        state = self.state

        power = min(abs(required_power), self.parameters.nominal_power)# [kW] how much electricity can be absorbed by the fuel cell absorb
        
        FC_hyd = power / self.h2p_el_eff_in            # [kW] hydrogen power input
        hyd = power/(self.h2p_el_eff_in*c.HHVH2*1000)    # [kg/s] amount of needed hydrogen for the given input power
        FC_Heat = FC_hyd * self.h2p_th_eff_in           # [kW] thermal power output

        water = (hyd*(self.h2oMolMass/self.H2MolMass))/self.rhoStdh2o       # [Sm3/s] stoichiometric water production
       
        etaFC = self.h2p_el_eff_in
        if hyd > available_hydrogen: # if available hydrogen is not enough to meet demand
            hyd     = available_hydrogen
            power   = hyd * self.h2p_el_eff_in 
            FC_Heat = hyd * self.h2p_th_eff_in
            water   = (hyd*(self.h2oMolMass/self.H2MolMass))/self.rhoStdh2o       # [Sm3/s] stoichiometric water production
            
        return (-hyd,power,FC_Heat,etaFC,water) # return hydrogen absorbed [kg] and electricity required [kW]
