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
        
class PemFuelCell(FuelCell):
    def __init__(self, parameters: FuelCellParameters, timestep_number: int, timestep = False):
        """Create a fuel cell object.
        Absorbs hydrogen and produces electricity, heat and water.
        See self#use

        Parameters
        ----------
        parameters: FuelCellParameters
            All the parameters the fuel cell needs to be instantiated
            NOTE: parameters.nominal_power needs to be lower than 1000

        timestep_number: int
            The number of timesteps the fuel cell should go through

        timestep: boolean = False
            TODO

        Raises
        ------
        ValueError
            If parameters.nominal_power >= 1000 
        """
        if self.parameters.nominal_power >= 1000:
            raise ValueError(f"Warning: {self.Npower} kW of rated power has been selected for the single PEM fuel cell module. \n\
            The maximum capacity is 1000 kW.\n\
            Options to fix the problem: \n\
                (a) -  Global fuel cell capacity can be increased by adding more modules in fuel cell parameters in studycase.json")

        super().__init__(parameters, timestep_number, timestep)

        # The general model has been vaidated with experimental data for the specifications below.  
        'Model: NEDSTACK FCS 13-XXL'      # https://nedstack.com/sites/default/files/2022-07/nedstack-fcs-13-xxl-gen-2.9-datasheet-rev01.pdf
        'Rated nominal power:  13.6 kW'
        # The model has then been adapted to scale main parameters as a function of the selected size. Maximum moule size = 1000 kW.
        # Key aspects of the model can be found in Chavan (2017): https://doi.org/10.1016/j.energy.2017.07.070)

        self.EFF                = np.zeros(timestep_number)    # [-]  Keeping track of fuel cell efficiency
        self.VOLT               = np.zeros(timestep_number)    # [V]  Keeping track of single cell working voltage - necessary for ageing calculations
        self.CURR_DENS          = np.zeros(timestep_number)    # [A]  Keeping track of single cell working current - necessary for ageing calculations
        self.EFF_last_module    = np.zeros(timestep_number)    # [-]  Keeping track of the elecrolyzer last module efficiency over the simulation
        self.n_modules_used     = np.zeros(timestep_number)    # [-]  Number of modules active at each timestep 

        "H2 --> 2H+ + 2e" 

        # FuelCell  Parameters 
        self.Lambda              = 23                                  # [-]      Cell mositure content
        self.FC_AnodeCurrDens    = 0.000000009                         # [A/cm^2] Anode current density
        self.FC_CathodeCurrDens  = 0.001                               # [A/cm^2] Cathode current density
        self.FC_OperatingTemp    = 273.15 + 62                         # [K]      https://www.google.com/search?q=working+temperature+PEM+fuel+cell&oq=working+temperature+PEM+fuel+cell+&aqs=chrome..69i57j0i512l9.10383j0j7&sourceid=chrome&ie=UTF-8
        self.FC_FuelPress        = 101325 + 25000                      # [Pa]     Fuel supply pressure (Anode) 
        self.FC_AirPress         = 101325                              # [Pa]     Air supply pressure (Cathode)
        self.FC_MinCurrDens      = 0.0001                              # [A/cm^2] FC min. current density      
        self.CTC                 = 0.4                                 # [-]      Charge Transfer Coefficient  - if Nominal Power < 6 kW: CTC = 0.45
                                                                       #                                  - if //   //   //  > 6 kW: CTC = 0.4
        self.MembThickness       = 250                                 # [μm]     Fuel cell membrane thickness - if Nominal Power < 6 kW: MembThickness = 100 
                                                                       #                                    - if //   //   //  > 6 kW: MembThickness = 145 
        self.FC_MaxCurrent       = 230                                 # [A]      Value taken from datasheet. Current value at which maximum power is delivered

        self.MaxPowerStack       = self.n_modules*self.Npower          # [kW]     Stack maximum power output

        self.nc                  = 90 + int((self.Npower/1000)*(250-90))    # For a power range between 0 kW and 1000 kW the number of cells in the stack varies between 90 and 250 
        self.FC_MaxCurrDens      = 1.2 + (self.Npower/1000)*(1.3-1.2)       # For a power range between 0 kW and 1000 kW the maximum current density varies between 1.2 and 1.3 A/cm2 

        # Varying the number of cells, module efficiency remains unchanged

        # nc = 96 and FC_MaxCurrDens = 1.2 are the values derived from the above-mentioned datasheet 
        # Such values should be used when implementing a fuel cell of 13.6 kW

        'POLARIZATION CURVE'

        Ndatapoints= 1000                    # Number of points used to compute the polarization curve 

        # Saving different losses contruibutions to be considered in the polarization curve

        self.OCpotential = []                # [V] Open circuit voltage
        self.ActLosses   = []                # [V] Activation losses
        self.OhmLosses   = []                # [V] Ohmic losses
        self.CellVolt = np.zeros(Ndatapoints)
        self.CellCurrDensity = np.linspace(self.FC_MinCurrDens,self.FC_MaxCurrDens,Ndatapoints)   

        'Polarization (V-i) curve calculation'

        # V is obtained by summing 3 different contributions (concentration losses are considered to be negligible)

        '1- Cell open curcuit voltage'

        pO2 = (self.FC_AirPress*0.21)/101325  # [atm] accounting for partial pressure of oxigen in air
        pH2O = 1                              # [atm]
        pH2 = self.FC_FuelPress/101325        # [atm]

        Ecell = 1.229 -0.85e-3*(self.FC_OperatingTemp-298.15) + 4.3085e-5*self.FC_OperatingTemp*ln(pH2*(pO2**0.5)/pH2O)  # [V] Open circuit voltage
        self.OCpotential = [Ecell for i in range(Ndatapoints)]
                        
        for i in range(0,Ndatapoints):
                                                    
            '2- Activation losses'
            Vact_cat =
                -self.Runiv * self.FC_OperatingTemp * np.log10(self.FC_CathodeCurrDens) / (self.CTC*4*self.FaradayConst)
                +self.Runiv * self.FC_OperatingTemp * np.log10(self.CellCurrDensity[i]) / (self.CTC*4*self.FaradayConst) #[V]



            Vact_an = -self.Runiv*self.FC_OperatingTemp*np.log10(self.FC_AnodeCurrDens)/(self.CTC*2*self.FaradayConst)+self.Runiv*self.FC_OperatingTemp*np.log10(self.CellCurrDensity[i])/(self.CTC*2.*self.FaradayConst)   #[V]

            Vact = Vact_cat +Vact_an               # [V] Activation losses
            
            self.ActLosses.append(Ecell-Vact)

            '3- Ohmic losses'
            rho_m = (181.6*(1+0.03*(self.CellCurrDensity[i])+0.062*((self.FC_OperatingTemp/303)**2)*(self.CellCurrDensity[i])**2.5))/ \
                ((self.Lambda-0.634-3*(self.CellCurrDensity[i]))*self.eNepero**(4.18*(self.FC_OperatingTemp-303)/self.FC_OperatingTemp))      #[Ohm*cm] specific membrane recistence
          
            Rm = rho_m*self.MembThickness/10000    # [Ohm/cm2]  Cell resistance depending on temperature and moisture content (Lambda) 

            Vohm = self.CellCurrDensity[i]*Rm      # [V] Ohmic losses
            
            self.OhmLosses.append(Ecell-Vact-Vohm)
            
            'SINGLE CELL VOLTAGE'
            
            self.CellVolt[i] = (Ecell-Vact-Vohm)                                  # [V]    Cell Voltage

        self.Vmin_FC=self.CellVolt[-1]*self.nc                                    # [V]    Minimum value for working voltage
        self.FC_CellArea=self.Npower*1000/(self.Vmin_FC*self.FC_MaxCurrDens)      # [cm^2] FC cell active area

        'RESULTING MODULE VOLTAGE'

        self.Voltage = self.CellVolt*self.nc                                      # [V] Module Voltage

        'Interpolation of  polarization curve: defining the fit-function for i-V curve'

        self.num = Ndatapoints                                                    # [-] number of intervals to be considered for the interpolation
        self.x = np.linspace(self.CellCurrDensity[0],self.CellCurrDensity[-1],self.num) 

        # Interpolating functions
        self.iV1 = interp1d(self.CellCurrDensity,self.Voltage,bounds_error=False,fill_value='extrapolate')    # Linear spline 1-D interpolation - MOdule Voltage
        self.iV2 = interp1d(self.CellCurrDensity,self.CellVolt,bounds_error=False,fill_value='extrapolate')   # Linear spline 1-D interpolation - Cell Voltage

        # Creating the reverse curve IP - necessary to define the exact functioning point        
        self.Current = self.CellCurrDensity*self.FC_CellArea    # [A] Defining the current value: same both for the single cell and the full stack!

        # Defining Fuel Cell Max Power Generation
        'Fuel Cell Max Power Output'

        FC_power = []
        for i in range(len(self.Current)):
            pot=self.Current[i]*self.Voltage[i]/1000       # [kW] Power
            FC_power.append(pot)
        self.MinPower = min(FC_power)

        self.IP = interp1d(self.Current,FC_power,bounds_error=False,fill_value='extrapolate')          
        self.P  = []
        for i in range(len(self.Current)):
            power = (self.iV1(self.CellCurrDensity[i])*self.Current[i])/1000     # [kW] Resolving the equation system via interpolation
            self.P.append(power)                                                 # [kW] Output power values varying current
            
        self.PI=interp1d(self.P,self.Current,bounds_error=False,fill_value='extrapolate')  # Interpolating function returning Current if interrogated with Power 
        self.Pi=interp1d(self.P,self.CellCurrDensity,bounds_error=False,fill_value='extrapolate')  # Interpolating function returning Current density if interrogated with Power 
        self.PV=interp1d(self.P,self.Voltage,bounds_error=False,fill_value='extrapolate')  # Interpolating function returning Voltage if interrogated with Power 
        self.Pv=interp1d(self.P,self.CellVolt,bounds_error=False,fill_value='extrapolate')  # Interpolating function returning Voltage if interrogated with Power 

        'Single module electricity production'

        # Creation of lists of values required for interpolation functions
        hydrogen                = []
        water                   = []
        electricity_produced    = []
        self.eta_module         = []
        FC_Heat_produced        = []

        for i in range(len(FC_power)):
           
            p_required = FC_power[i]                                   # [kW] 
            FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea  # [A/cm^2] current density value at which the fuel cell is working 
            Current = FC_CellCurrDensity*self.FC_CellArea     # [A] FuelCell Stack operating current 
            
            FC_Vstack = self.iV1(FC_CellCurrDensity)          # [V] Stack operating voltage
            V_cell = FC_Vstack/self.nc                        # [V] Single cell operating voltage

            # Just backwards again: Pi -> Ii and Vi at current step i using the Area A

            'Computing FC efficiency and hydrogen energy demand'    
            
            pO2 = (self.FC_AirPress*0.21)/101325              # [atm]
            pH2O = 1                                          # [atm]
            pH2 = self.FC_FuelPress/101325                    # [atm]  
            
            Ecell = 1.229-0.85e-3*(self.FC_OperatingTemp-298.15) + 4.3085e-5*self.FC_OperatingTemp*ln(pH2*(pO2**0.5)/pH2O)   # [V] Open circuit voltage 
            deltaG = self.deltaG0 - self.Runiv*self.FC_OperatingTemp*ln(pH2*math.sqrt(pO2)/pH2O)/(2*self.FaradayConst)       # [kJ/mol] Gibbs free energy at actual conditions
          
            eta_voltage = FC_Vstack/(Ecell*self.nc)           # [-] Voltage efficiency 
            eta_th = - deltaG/self.HHVh2Mol                   # [-] Thermodynamic efficiency
         
            etaFC = eta_th*eta_voltage                        # [-] FC efficiency

            'Hydrogen demand'
         
            FC_HydroCons = (Current*self.nc/95719.25)/1000/self.rhoStdh2    # [kg/s]*[Sm3/kg] = [Sm3/s] (Chavan 2017)
            hyd = FC_HydroCons*self.rhoStdh2                                # [kg/s]
            FC_deltaHydrogen = - hyd*self.HHVh2*1000                        # [kW]
            
            'Water production'
            # water_produced = (hyd*self.h2oMolMass/self.H2MolMass)/self.rhoStdh2o            # [Sm3/s] stoichiometric amount
            water_produced = (((p_required*1000/(V_cell*2*self.FaradayConst))*self.h2oMolMass)/self.rhoStdh2o)*self.nc # [Sm3/s] module produced water floe rate https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118878330.app2
            
            'Process heat, that can be recovered'
          
            heat_loss = 0.2 * (-FC_deltaHydrogen)      # Assuming 20% of energy losses                                                                            
            FC_Heat = ((1.481*self.nc)/FC_Vstack-1)*p_required -  heat_loss              # [kW] 
                
            hydrogen.append(hyd)                        # [kg/s]    consumed hydrogen
            water.append(water_produced)                # [m^3/s]   produced water
            electricity_produced.append(p_required)     # [kW]      output power
            self.eta_module.append(etaFC)               # [-]       fc efficiency
            FC_Heat_produced.append(FC_Heat)            # [kW]      co-product heat

        self.max_h2_module  = max(hydrogen)                         # [kg/s] maximum amount of exploitable hydrogen for the considered module
        self.max_h2_stack   = self.max_h2_module*self.n_modules     # [kg/s] maximum amount of exploitable hydrogen for the considered stack
        self.maxVolt_module = max(self.Voltage)                     # [V] maximum voltage of the considered module
        self.minVolt_module = min(self.Voltage)                     # [V] minimum voltage of the considered module

        electric_eff_module = self.eta_module[-1]  # Module electric efficiency
        h2p_eff = (1/(electric_eff_module*c.HHVH2*1000))*3600 # [kg/kWh] kg/h of hydrogen produced per kWh of input power
        thermal_eff_module = (FC_Heat_produced[-1] / - FC_deltaHydrogen) # Module thermal efficiency
        total_efficiency = electric_eff_module + thermal_eff_module

        print(f"\nThe fuel cell electric efficiency of each module is found to be equal to {electric_eff_module*100:.2f}%, which is equivalent to {round(h2p_eff, 3)} kg/kWh (using H2 HHV). "
              f"The fuel cell thermal efficiency of each module is found to be equal to {thermal_eff_module*100:.2f}% with heat output available for cogeneration at {self.FC_OperatingTemp}K. "
              f"Thus, the total efficiency is equal to {total_efficiency*100:.2f}%. "
              f"A fuel cell nominal power of {self.MaxPowerStack:.2f} kW needs to be fed with {self.max_h2_stack*3600:.2f} kg/h of hydrogen.")
        self.etaFuelCell = interp1d(hydrogen,self.eta_module,bounds_error=False,fill_value='extrapolate')          # Linear spline 1-D interpolation -> H2 consumption - FC efficiency
        self.h2P         = interp1d(hydrogen,electricity_produced,bounds_error=False,fill_value='extrapolate')  # Linear spline 1-D interpolation -> H2 consumption - produced electricity
        self.FC_Heat     = interp1d(hydrogen,FC_Heat_produced,bounds_error=False,fill_value='extrapolate')      # Linear spline 1-D interpolation -> H2 consumption - produced heat
        self.water       = interp1d(hydrogen,water,bounds_error=False,fill_value='extrapolate')                 # Linear spline 1-D interpolation -> H2 consumption - produced water

        self.iEta   = interp1d(self.CellCurrDensity,self.eta_module,bounds_error=False,fill_value='extrapolate')       # Linear spline 1-D interpolation -> Operating current density - efficiency
        self.IEta   = interp1d(self.Current,self.eta_module,bounds_error=False,fill_value='extrapolate')       # Linear spline 1-D interpolation -> Operating current density - efficiency
        self.ihyd   = interp1d(self.CellCurrDensity,hydrogen,bounds_error=False,fill_value='extrapolate')           # Linear spline 1-D interpolation -> Operating current density - H2 consumption
        self.Ihyd   = interp1d(self.Current,hydrogen,bounds_error=False,fill_value='extrapolate')           # Linear spline 1-D interpolation -> Operating current density - H2 consumption
        self.iHeat  = interp1d(self.CellCurrDensity,FC_Heat_produced,bounds_error=False,fill_value='extrapolate')   # Linear spline 1-D interpolation -> Operating current density - produced heat            
        self.IHeat  = interp1d(self.Current,FC_Heat_produced,bounds_error=False,fill_value='extrapolate')   # Linear spline 1-D interpolation -> Operating current density - produced heat            
        self.iwater = interp1d(self.CellCurrDensity,water,bounds_error=False,fill_value='extrapolate')              # Linear spline 1-D interpolation -> Operating current density - produced water            
        self.Iwater = interp1d(self.Current,water,bounds_error=False,fill_value='extrapolate')              # Linear spline 1-D interpolation -> Operating current density - produced water            

        if self.ageing:             # if ageing effects are being considered
            self.stack = {
                            'Activation[-]': np.zeros(timestep_number),                    # Initialize an array to track module activation (1 for on, 0 for off) for each timestep
                            'Pol_curve_history': [],                                       # Initialize an empty list to keep track of polarization curve shifts during utilization
                            'Module_efficiency[-]': [],                                    # Initialize an empty list to keep track of module efficiency over time
                            'Conversion_ratio_op[kWh/kg]': np.zeros(timestep_number),        # Initialize an array to keep track of performance evolution
                            'Conversion_ratio_rated[kWh/kg]': np.zeros(timestep_number),        # Initialize an array to keep track of performance evolution
                            'hydrogen_consumption[kg/s]': np.zeros(timestep_number),        # Initialize an array to keep track of hydrogen production
                            'i_op[A]': np.zeros(timestep_number),                          # Initialize an array to keep track of operating current
                            'v_op[V]': np.zeros(timestep_number)                           # Initialize an array to keep track of operating voltage
                            }
            self.Γ              = (self.Npower)/(self.max_h2_module*3600) # [kWh/kg] ideal coversion ratio
            # Defining the optimal operating range
            self.v_0        = self.minVolt_module/self.nc    # [V] minimun voltge for the single cell
            self.vol_max    = self.maxVolt_module/self.nc    # [V] maximum voltge for the single cell
            self.v_L        = 0.6*self.vol_max               # [V] lower boundary of the optimal range 
            self.v_U        = 0.8*self.vol_max               # [V] upper boundary of the optimal range      
            admissile_loss  = 20                             # [%] admissible voltage values loss compared to rated performance
            self.CellVoltage_limit = max(self.Voltage)*(admissile_loss) # [V] cell voltage value requiring replacement of the module at end of life
            self.polarization_curve_ageing = self.Voltage.copy()  # [V] initialising pol_curve. Considering design performances at first step (before starting degradation computing)
            self.hydP       = interp1d(hydrogen,self.P)
            self.hydcons    = hydrogen

    def plot_polarizationpts(self):
        fig=plt.figure(figsize=(10,8),dpi=1000)
        fig.suptitle("PEMFC Polarization Curve STACK - P ={}".format(round(self.Npower,1)) +" kW")

        plt.plot(self.CellCurrDensity,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        plt.plot(self.x,self.iV1(self.x),label='linear interp',linestyle='--') 
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlabel('Cell Current Density [A cm$^{-2}$]')
        plt.ylabel('Stak Voltage [V]')
        plt.title('PEMFC Polarization Curve (V-i)' )

        fig=plt.figure(figsize=(8,8),dpi=1000)
        fig.suptitle("PEMFC Polarization Curve STACK - P ={}".format(round(self.Npower,1)) +" kW")

        ax_1=fig.add_subplot(221)
        ax_1.plot(self.CellCurrDensity,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        ax_1.plot(self.x,self.iV1(self.x),label='linear interp',linestyle='--') 
        ax_1.grid()
        ax_1.legend(fontsize=8)
        ax_1.set_xlabel('Cell Current Density [A cm$^{-2}$]')
        ax_1.set_ylabel('Stak Voltage [V]')
        ax_1.set_title('PEMFC Polarization Curve (V-i)' )

        ax_2=fig.add_subplot(222)
        ax_2.plot(self.Current,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        #ax_2.plot(self.x,self.iV1(self.x),label='linear',linestyle='--') 
        ax_2.grid()
        ax_2.legend(fontsize=8)
        ax_2.set_xlabel('Cell Current [A]')
        ax_2.set_ylabel('Stak Voltage [V]')
        ax_2.set_title('PEMFC Polarization Curve (V-I)' )

        plt.tight_layout()
        plt.show()


        'Polarization Curve'

        plt.figure(dpi=600, figsize=(9,5))
        plt.plot(self.CellCurrDensity,self.CellVolt,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        plt.plot(self.CellCurrDensity,self.OCpotential,label='Open Circuit Potential')
        plt.plot(self.CellCurrDensity,self.ActLosses,label='Activation Losses')
        plt.plot(self.CellCurrDensity,self.OhmLosses,label='Ohmic Losses')
        plt.plot(self.x,self.iV2(self.x),label='linear interp',linestyle='--') 
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlabel('Cell Current Density [A cm$^{-2}$]')
        plt.ylabel('Stak Voltage [V]')
        plt.title('PEMFC Polarization Curve (V-i)' )

    def plot_stackperformance(self):
        'Datasheet FCS 13-XXL Gen 2.9'

        Current      = [0,40,80,120,160,200,230,250]
        StackVoltage = [94,78,73,69,66,62,59,57]
        StackPower   = [0,3.1,5.8,8.3,10.5,12.4,13.6,14.1]
        x = np.linspace(0,250,100)

        interp = interp1d(Current,StackVoltage,bounds_error=None,kind='cubic',fill_value='extrapolate')
        interp1= interp1d(Current,StackPower,bounds_error=None,kind='cubic',fill_value='extrapolate')

        'Polarization Curve'

        fig,ax = plt.subplots(dpi=600,figsize=(9,5))
        ax2 =ax.twinx()
        ax.plot(self.Current,self.Voltage,label='V$_\mathregular{stack}$ Model')
        ax.plot(Current,StackVoltage,linestyle='None',marker='.',mec='r',markersize=10)
        ax.plot(x,interp(x),label='V datasheet',marker='.', markersize=2)
        ax.axvline(x=230,linestyle='--',color='k')
        ax2.axhline(y=13.6,linestyle='--',color='k')
        ax.grid()
        ax2.plot(self.Current,self.P,label='P$_\mathregular{stack}$ Model',color='tab:red')
        ax2.plot(Current,StackPower,linestyle='None', marker='.',mec='b',markersize=10)
        ax2.plot(x,interp1(x),label='P datasheet',color='tab:orange',marker='.', markersize=2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=4)
        ax.set_ylim(0,None)
        ax.set_xlabel('Current [A]')
        ax.set_ylabel('Stak Voltage [V]')
        ax2.set_ylabel('Stack Power [kW]')       
        plt.title('PEMFC STACK Performance (V-I)')

    def plot_linregression(self): 
        'Linear Regression'
        x1 = self.CellCurrDensity.reshape((-1,1))
        y1 = self.Voltage

        model = LinearRegression().fit(x1,y1)
        r_sq_linreg = model.score(x1,y1)
        print('Coeff. of Determination:', r_sq_linreg)

        self.coeff_A = model.intercept_      # Obtaining the calculated intercept for the linear fit - returns a scalar
        print('intercept:', self.coeff_A)
        self.coeff_B = model.coef_           # Obtaining the calculated slope for the linear fit - returns an array with only 1 value
        print('slope:', self.coeff_B)

        Volt_LinReg = self.coeff_A + self.coeff_B*self.CellCurrDensity

        # i-V plot Linear Regression 

        fig=plt.figure(figsize=(8,8),dpi=1000)
        fig.suptitle("Prestazioni FC da {}".format(round(self.NPower,1)) +" kW")

        ax_1=fig.add_subplot(121)
        ax_1.plot(self.CellCurrDensity,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        ax_1.plot(self.CellCurrDensity,Volt_LinReg,label='Least Squares LinReg') 
        ax_1.grid()
        ax_1.legend(fontsize=8)
        ax_1.set_xlabel('Cell Current Density [A cm$^{-2}$]')
        ax_1.set_ylabel('Stak Voltage [V]')
        ax_1.set_title('PEMFC Polarization Curve (V-i)' )

        ax_2=fig.add_subplot(122)
        ax_2.plot(self.Current,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        #ax_2.plot(self.Current,Volt_LinReg,label='Least Squares LinReg') 
        ax_2.grid()
        ax_2.legend(fontsize=8)
        ax_2.set_xlabel('Cell Current [A]')
        ax_2.set_ylabel('Stak Voltage [V]')
        ax_2.set_title('PEMFC Polarization Curve (V-I)' )

        plt.tight_layout()
        plt.show()

    def use(self, step: int, required_power: float, available_hyd: float):
        super().use(step, required_power, available_hyd)

        available_hydrogen = self.available_hydrogen
        state = self.state

        # PowerOutput [kW] - Electric Power required from the fuel cell
        if (abs(required_power) <= self.Npower) or (available_hydrogen/self.max_h2_module < 1):
            # Required power higher than the nominal power of the module
            #   or available hydrogen lower that the max flow one module uses

            # Only one module
            # Produces required_power but at least the min power of one module

            p = max(abs(required_power), self.MinOutputPower)

            hyd,power,FC_Heat,etaFC,water = self.use1(self,step,-p,available_hydrogen)
            if abs(hyd) > 0:
                self.n_modules_used[step] = 1
            else:
                self.n_modules_used[step] = 0
                    
            self.EFF[step]   = etaFC

        elif abs(required_power) > self.MaxPowerStack and available_hydrogen > self.max_h2_stack:
            # Full load

            hyd,power,FC_Heat,etaFC,water = self.use1(self,step,-self.Npower,available_hydrogen)
            
            hyd     = hyd*self.n_modules
            power   = power*self.n_modules
            FC_Heat = FC_Heat*self.n_modules
            water   = water*self.n_modules
            
            self.n_modules_used[step]   = self.n_modules
            self.EFF[step]              = etaFC

        elif abs(required_power) > self.Npower:

            # Partial load: multiple modules

            # number of required modules working at full load
            required_full_modules = min(self.n_modules, int(abs(required_power)/self.Npower))            

            # number of modules operating full load based on the amount of hydrogen available   
            full_modules = min(required_full_modules,int(available_hydrogen/self.max_h2_module))  

            hyd,power,FC_Heat,etaFC_full,water = self.use1(self,step,-self.Npower,available_hydrogen)
            
            hyd_full        = hyd*full_modules
            power_full      = power*full_modules
            FC_Heat_full    = FC_Heat*full_modules
            water_full      = water*full_modules
            
            residual_power      = abs(required_power) - power_full 
            residual_hydrogen   = available_hydrogen - abs(hyd_full)

            if residual_power >= self.MinOutputPower:
                hyd_singlemodule,power_singlemodule,FC_Heat_singlemodule,etaFC_singlemodule,water_singlemodule = self.use1(self,step,-residual_power,residual_hydrogen)

                if hyd_singlemodule != 0:   # single module operating in partial load 
                    self.n_modules_used[step] = full_modules + 1
                    self.EFF[step] = ((full_modules*etaFC_full) + (etaFC_singlemodule))/self.n_modules_used[step]  # weighted average 
                    self.EFF_last_module[step] = etaFC_singlemodule
                else: 
                    self.n_modules_used[step] = full_modules
                    self.EFF[step] = etaFC_full
            else:
                # MISTAKE? => should be just residual_power
                residual_power = 0
                hyd_singlemodule,power_singlemodule,FC_Heat_singlemodule,water_singlemodule = [0]*4
                self.n_modules_used[step] = full_modules
                self.EFF[step] = etaFC_full
                
            hyd     = hyd_full + hyd_singlemodule           # [kg/s]  produced hydrogen   
            power   = power_full + power_singlemodule       # [kW] output power
            FC_Heat = FC_Heat_full + FC_Heat_singlemodule   # [kW] co-product heat
            etaFC   = self.EFF[step]
            water   = water_full + water_singlemodule       # [m^3/s] produced water
            
        return (hyd,power,FC_Heat,etaFC,water)  # return hydrogen absorbed [kg/s] electricity required [kW] and heat [kW] and water [Sm3] as a co-products 

    def use1(self, step: int, required_power: float, available_hydrogen: float):
        """Single module operation.
        Finding the working point of the FuelCell by explicitly solving the system:

        Parameters
        ----------
        step: int
            The step to be simulated (an index to the operational_period)

        required_power: float
            The power in kW the system requires at the given step.
            Value will be used as abs, so it does not matter if the input is positive or negative.

        available_hydrogen: float
            The hydrogen in kg/s that can be supplied by the system at the current step

        Returns
        -------
        tuple[float, float, float, float]
            0: hydrogen consumption [kg/s]
            1: electricity [kW] 
            2: heat [kW]
            3: water [Sm^3]
        """
        p_required = abs(required_power)                                             # [kW] 
        FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea   # [A/cm^2] current density value at which the fuel cell is working
        Current = FC_CellCurrDensity*self.FC_CellArea               # [A] Fuel Cell module operating current
        # Checking if resulting current density is high enough for the fuel cell to start, otherwise hydrogen used = 0
        if FC_CellCurrDensity < self.FC_MinCurrDens or p_required < self.MinOutputPower:
            # Either the current density is lower than possible
            #  or the required_power is lower that possible
            etaFC               = 0     # [-] fuel cell efficiency
            hyd                 = 0     # [kg/s] hydrogen used in the considered timestep
            Current             = 0     # [A] Operational Current
            p_required          = 0     # [kW] required energy - when timestep is kept at 1 h kW = kW
            FC_Heat             = 0     # [kW] thermal energy produced
            FC_CellCurrDensity  = 0     # [A/cm2] fuel cell current density
            water               = 0     # [Sm3/s] water production
            if self.ageing:
                hyd,p_required,FC_Heat,etaFC,water = fuel_cell.ageing(self,step,p_required)
        else:
            if self.ageing:
                hyd,p_required,FC_Heat,etaFC,water = fuel_cell.ageing(self,step,p_required)
            else:
                V_cell      = self.iV2(FC_CellCurrDensity)          # [V] Single cell opertaing voltage 
                FC_Vstack   = self.iV1(FC_CellCurrDensity)          # [V] Module operating voltage

                self.VOLT[step]      = V_cell                       # [V]      Cell voltage history
                self.CURR_DENS[step] = FC_CellCurrDensity           # [A/cm2]  Cell current density history
             
                etaFC       = float(self.iEta(FC_CellCurrDensity))         # [-] FC efficiency                
                hyd         = float(self.ihyd(FC_CellCurrDensity))         # [kg/s] hydrogen consumed  
                FC_Heat     = float(self.iHeat(FC_CellCurrDensity))        # [kW] thermal energy produced
                water       = float(self.iwater(FC_CellCurrDensity))       # [Sm3/s] water production 

                if hyd > available_hydrogen:     # if not enough hydrogen is available in the system to meet demand (H tank is nearly empty)
                    # defining the electric load that can be covered with the hydrogen available 
                    hyd,p_required,FC_Heat,etaFC,water = fuel_cell.h2power(self,available_hydrogen)

        return (-hyd,p_required,FC_Heat,etaFC,water)  # return hydrogen absorbed [kg/s] electricity required [kW] and heat as a co-product [kW]
