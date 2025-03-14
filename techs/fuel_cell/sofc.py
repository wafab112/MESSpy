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

class SofcFuelCell(FuelCell):
    def __init__(self, parameters: FuelCellParameters, timestep_number: int, timestep = False):
        if self.parameters.nominal_power >= 1000:
            raise ValueError(f"Warning: {self.Npower} kW of rated power has been selected for the single SOFC fuel cell module. \n\
            The maximum capacity is 1000 kW.\n\
            Options to fix the problem: \n\
                (a) -  Global fuel cell capacity can be increased by adding more modules in fuel cell parameters in studycase.json")
        super().__init__(parameters, timestep_number, timestep)

        self.EFF=np.zeros(timestep_number)                    # [-]      keeping track of fuel cell efficiency
        self.EFF_last_module     = np.zeros(timestep_number)  # [-]      keeping track of the elecrolyzer last active module efficiency over the simulation
        self.n_modules_used      = np.zeros(timestep_number)  # [-]      Number of modules active at each timestep 
        
        self.FC_OperatingTemp             = 273.15+800         # [K]      Operating temperature
        self.FC_RefTemp                   = 273.15+750         # [K]      Operating temperature reference
        self.FC_FuelPress                 = 116000 #506625             # [Pa]     Fuel supply pressure (Anode)  
        self.FC_AirPress                  = 101325/101325 #444407/101325      # [atm]    Air supply pressure (Cathode)
        # self.FC_MinCurrDens               = 0.0001              # [A/cm^2] FC min. current density - arbitrary
        self.FC_MinCurrDens               = 0.04194            # [A/cm^2] FC min. current density - derived from experimental plot

        self.FC_AlfaAnode                 = 0.55               # [-]      Charge transfer coefficient for anode
        self.FC_AlfaCathode               = 0.33               # [-]      Charge transfer coefficient for cathode 
        self.FC_ActivationEnergyAnode     = 110                # [kJ/mol] Anonde activation energy  -> ref.:http://dx.doi.org/10.1016/j.desal.2017.02.013
        self.FC_ActivationEnergyCathode   = 160                # [kJ/mol] Cathode activation energy -> ref.: //        //        //        //        //
        self.FC_ActivationCoeff           = 10                 # [-]      Activation coefficient 
        self.FC_ExchangeCurrDensChAn      = 0.530              # [A/cm^2] Exchange current density channel anode 
        self.FC_ExchangeCurrDensChCat     = 0.200              # [A/cm^2] Exchange current density channel cathode 
        self.FC_ExchangeCurrDensCathode   = self.FC_ExchangeCurrDensChCat*self.eNepero**(((self.FC_ActivationCoeff*self.FC_ActivationEnergyCathode)\
                                            /self.Runiv)*((1/self.FC_RefTemp)-(1/self.FC_OperatingTemp)))                                   # [mA/cm^2]
        self.FC_ModFactor                 = 2                  # [-]      Ohmic losses modification factor 
        self.ThicknessElectrolyte         = 0.00003 #0.00006            # [m]      Electrolyte thickness
        self.ElectrolyteCostant           = 50                 # [K/ohm*m]
        self.ActivationEnergyElectrolyte  = 9*10**(7)          # [kJ/mol] Electrolyte activation energy
        self.FC_LimitCurrDens             = 6                  # [A/cm^2]
        self.DiffusionVolumeH20           = 13.1               # [-] 
        self.DiffusionVolumeH2            = 6.1                # [-]      
        self.stoichiometriccoeff          = 2                  # [-]    coefficient used to account for working in excess air

        self.MaxPowerStack      = self.n_modules*self.Npower       # [kW]     Max power output 
        
        self.nc                  = 80 + int((self.Npower/1000)*(250-80))       # For a power range between 0kW and 1000kW the number of cells in the stack varies between 80 and 250 
        self.FC_MaxCurrDens      = 1.25 + (self.Npower/1000)*(1.25-1.25)          # For a power range between 0kW and 1000kW the maximum current density can be varied between the chosen values A/cm2 
        
        # Varying the number of cells, module efficiency remains unchanged
        # nc = 77 and FC_MaxCurrDens = 1.54 are derived from the above-mentioned datasheet
        
        'POLARIZATION CURVE'
        
        Ndatapoints = 1000                        # Number of points used to compute the polarization curve 
        
        self.DeltaV_ohm = np.zeros(Ndatapoints)   # Ohmic losses
        self.DeltaV_con = np.zeros(Ndatapoints)   # Concentration losses
        self.Ecell = []                           # Open circuit voltage
        self.CellVoltage = np.zeros(Ndatapoints)
        self.CellCurrDensity = np.linspace(self.FC_MinCurrDens,self.FC_MaxCurrDens,Ndatapoints)  
        
        pH2 = self.FC_FuelPress/101325                  # [atm]
        pH2O = 1                                        # [atm]
        pO2 = (self.FC_AirPress*0.21*101325)/101325     # [atm]
        
        # V is obtained by summing 4 different contributions 
        'Polarization (V-i) curve calculation'
    
        '1 - Open circuit potential'                
        
        Ecell = 1.19 + ((self.Runiv*self.FC_OperatingTemp)/(2*self.FaradayConst))*ln(pH2*(pO2**0.5)/pH2O)  # [V] Open circuit voltage
        self.Ecell  = [Ecell for i in range(Ndatapoints)]

        for i in range(0,Ndatapoints):
                            
            '2 - Ohmic losses'
            
            self.DeltaV_ohm[i] = (self.CellCurrDensity[i]*self.ThicknessElectrolyte*self.FC_OperatingTemp)/\
                                 (9000*self.eNepero**(-100000/(self.Runiv*self.FC_OperatingTemp)))                   # [V] (100000 activation energy in [kJ/mol], 9000 electrolyte constant)
           
            '3 - Concentration losses'   # Significant losses only for cathode, activation losses (minimal contribution for SOFC) are also present in the following formula
            
            self.DeltaV_con[i] = ((self.Runiv*self.FC_OperatingTemp)/(2*self.FaradayConst))*\
                                 ln((self.CellCurrDensity[i]/(self.FC_ExchangeCurrDensCathode*self.FC_AirPress*(0.21-0.0008*10000*self.CellCurrDensity[i]*self.Runiv*self.FC_OperatingTemp/(4*self.FaradayConst*101325*0.00002)))))        # [V] 
            
            '4 - Cell voltage'
            
            self.CellVoltage[i] = self.Ecell[i]-self.DeltaV_ohm[i]-self.DeltaV_con[i]
            
        self.Vmin_FC_stack = self.CellVoltage[-1]*self.nc                                 # [V]    Minimum value for working voltage
        self.FC_CellArea = self.Npower*1000/(self.Vmin_FC_stack*self.FC_MaxCurrDens)      # [cm^2] FC cell active area
        self.FC_Pmax = self.Npower                                                        # [kW] Max output power
        
        '5- Stack voltage'
        
        self.Voltage = self.nc*self.CellVoltage

        'Interpolation of  polarization curve: defining the fit-function for i-V curve'    
      
        self.num = Ndatapoints                                          # [-] number of intervals to be considered for the interpolation
        self.x = np.linspace(self.CellCurrDensity[0],self.CellCurrDensity[-1],self.num)
                
        # Defining different interpolation methods
        self.iV1 = interp1d(self.CellCurrDensity,self.Voltage,bounds_error=False,fill_value='extrapolate')          # Linear spline 1-D interpolation
        
        # Creating the reverse curve IP - necessary to define the exact functioning point
        self.Current = self.CellCurrDensity*self.FC_CellArea

        'SOFC Max Power Production'
        
        FC_power = []
        for i in range(len(self.Current)):
             pot = self.Current[i]*self.Voltage[i]/1000     # [kW] Power
             FC_power.append(pot)
        self.MinPower = min(FC_power)
        
        self.IP=interp1d(self.Current,FC_power,kind='cubic',bounds_error=False,fill_value='extrapolate') 
        self.P = np.zeros(Ndatapoints)
        
        for i in range (len(self.Current)):
              self.P[i] = (self.iV1(self.CellCurrDensity[i])*self.Current[i])/1000   # [kW] power output 
        
        self.PI=interp1d(self.P,self.Current,bounds_error=False,fill_value='extrapolate')   # Interpolating function returning Current if interrogated with Power 

        'Single module electricity production'
        # Creation of lists of values required for interpolation functions
        hydrogen                = []
        water                   = []
        electricity_produced    = []
        eta_FuelCell            = []
        FC_Heat_produced        = []

        for i in range(len(FC_power)):
            
            p_required = FC_power[i]                                    # [kW] power production               
            FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea   # [A/cm^2] current density value at which the fuel cell is working 
            Current = FC_CellCurrDensity*self.FC_CellArea               # [A] FuelCell Stack operating current
            FC_Vstack= self.iV1(FC_CellCurrDensity)                     # [V] Stack operating voltage

            'Computing FC efficiency and hydrogen energy demand'    
            
            pO2 = (self.FC_AirPress*0.21*101325)/101325      # [atm]
            pH2O = 1                                         # [atm]
            pH2 = self.FC_FuelPress/101325                   # [atm]  
            
            Ecell = 1.19 + ((self.Runiv*self.FC_OperatingTemp)/(2*self.FaradayConst))*ln(pH2*(pO2**0.5)/pH2O)
            DeltaV_OHM = (FC_CellCurrDensity*self.ThicknessElectrolyte*self.FC_OperatingTemp)/\
                (9000*self.eNepero**(-100000/(self.Runiv*self.FC_OperatingTemp)))
            DeltaV_CON = ((self.Runiv*self.FC_OperatingTemp)/(2*self.FaradayConst))\
                *ln((FC_CellCurrDensity/(self.FC_ExchangeCurrDensCathode*self.FC_AirPress*(0.21-0.0008*10000*FC_CellCurrDensity*self.Runiv*self.FC_OperatingTemp/(4*self.FaradayConst*101325*0.00002)))))
            TotalLoss = DeltaV_OHM + DeltaV_CON   # [V]
            
            deltaG = self.deltaG0 - self.Runiv*self.FC_OperatingTemp*ln(pH2*math.sqrt(pO2)/pH2O)/(2*self.FaradayConst)    # [kJ/mol] Gibbs free energy at actual conditions
          
            eta_voltage = FC_Vstack/(Ecell*self.nc)     # [-] Voltage efficiency 
            eta_th = self.deltaG0/self.HHVh2Mol         # [-] Thermodynamic efficiency
            etaFC = -eta_th*eta_voltage                 # [-] FC efficiency
            
            'Hydrogen demand'
            FC_HydroCons = ((Current*self.nc)/(self.FaradayConst*1000))/(self.rhoStdh2)     # [kg/s]*[Sm3/kg] = [Sm3/s] 
            hyd = FC_HydroCons*self.rhoStdh2                                                # [kg/s]
            FC_deltaHydrogen = - hyd*self.HHVh2*1000                                        # [kW]
            
            'Air demand'     
            FC_AirCons       = ((self.AirMolMass*1000*p_required/\
                                 (self.FaradayConst*2*FC_Vstack/self.nc))*self.stoichiometriccoeff)         #[kg/s] air consumption
            FC_O2Cons        = FC_AirCons*0.2319                                                            #[kg/s] oxygen consumption, taken from the air 
            
            'Air exit flow rate'
            FC_AirExit = FC_AirCons-FC_O2Cons       #[kg/s] hourly outgoing air mass flow rate 
            
            'Heat Demand for air and hydrogen'                           
            Q_air=((FC_AirCons*self.cpAir*(self.FC_OperatingTemp-self.AmbTemp)))    #[kW] Heat needed to rise the temperatura of inlet air during ramp-up   (Q=m*c_p*DT)
            Q_h2=(hyd*self.cpH2*(self.FC_OperatingTemp-self.AmbTemp))                 #[kW] Heat needed to rise the temperature of inlet hydrogen during ramp-up (Q=m*c_p*DT) 
     
            'Water production'
            water_produced = (hyd*self.h2oMolMass/self.H2MolMass)/self.rhoStdh2o            # [m^3/s] stoichiometric amount
            # water_produced = ((p_required*1000/(V_cell*2*self.FaradayConst))*self.h2oMolMass)/self.rhoStdh2o # [Sm3/s] of produced water https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118878330.app2
            #!!! '[...] if methane is internally reformed, then half the product water is used in the reformation process, thus halving the rate of production https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118878330.app2
        
            'Steam produced'
            FC_H20Produced = hyd+FC_O2Cons                     #[kg/s] steam produced
            
            'Heat Produced by the electrochemical reaction'
            z = Current/(2*self.FaradayConst)   # [mol/s]
            DeltaS = -((self.H20MolStdEntropy-(self.O2MolStdEntropy/2)-self.H2MolStdEntropy)+\
                       (self.Runiv/2)*ln((pH2**2)*pO2/(pH2O**2)))                                   # [J/mol*K]
            FC_Heat_elec = ((((z*self.FC_OperatingTemp*DeltaS + Current*TotalLoss))*self.nc)/1000)  # [kW] 
            
            'Heat contained in anodic and cathodic flow'
            FC_HeatH20 = FC_H20Produced*self.SteamSH                                    #[kW] heat in the steam flow exiting the anode
            FC_HeatAir = FC_AirExit*self.cpAir*(self.FC_OperatingTemp-self.AmbTemp)     #[kW] heat in the air flow exiting the cathode
            FC_Heat    = FC_HeatH20+FC_HeatAir                                          #[kW] net heat available for cogeneration

            hydrogen.append(hyd)                       # [kg/s]  produced hydrogen
            water.append(water_produced)               # [m^3/s] produced water
            electricity_produced.append(p_required)    # [kW] output power
            eta_FuelCell.append(etaFC)                 # [-]   fc efficiency
            FC_Heat_produced.append(FC_Heat)           # [kW] co-product heat
            
        self.max_h2_module = max(hydrogen)                 # [kg/s] maximum amount of exploitable hydrogen for the considered module
        self.max_h2_stack   = self.max_h2_module*self.n_modules     # [kg/s] maximum amount of exploitable hydrogen for the considered stack
        
        electric_eff_module = eta_FuelCell[-1] # Module electric efficiency
        h2p_eff = (1/(electric_eff_module*c.HHVH2*1000))*3600 # [kg/kWh] kg/h of hydrogen produced per kWh of input power
        thermal_eff_module = (FC_Heat_produced[-1] / - FC_deltaHydrogen) # Module thermal efficiency
        total_efficiency = electric_eff_module + thermal_eff_module
        print(f"\nThe fuel cell electric efficiency of each module is found to be equal to {electric_eff_module*100:.2f}%, which is equivalent to {round(h2p_eff, 3)} kg/kWh (using H2 HHV). "
              f"The fuel cell thermal efficiency of each module is found to be equal to {thermal_eff_module*100:.2f}% with heat output available for cogeneration at {self.FC_OperatingTemp}K. "
              f"Thus, the total efficiency is equal to {total_efficiency*100:.2f}%. "
              f"A fuel cell nominal power of {self.MaxPowerStack:.2f} kW needs to be fed with {self.max_h2_stack*3600:.2f} kg/h of hydrogen.")

        self.etaFuelCell = interp1d(hydrogen,eta_FuelCell,bounds_error=False,fill_value='extrapolate')          # Linear spline 1-D interpolation -> H2 consumption - FC efficiency
        self.h2P         = interp1d(hydrogen,electricity_produced,bounds_error=False,fill_value='extrapolate')  # Linear spline 1-D interpolation -> H2 consumption - produced electricity
        self.FC_Heat     = interp1d(hydrogen,FC_Heat_produced,bounds_error=False,fill_value='extrapolate')      # Linear spline 1-D interpolation -> H2 consumption - produced heat
        self.water       = interp1d(hydrogen,water,bounds_error=False,fill_value='extrapolate')                 # Linear spline 1-D interpolation -> H2 consumption - produced water
        
        self.iEta   = interp1d(self.CellCurrDensity,eta_FuelCell,bounds_error=False,fill_value='extrapolate')       # Linear spline 1-D interpolation -> Operating current density - efficiency
        self.ihyd   = interp1d(self.CellCurrDensity,hydrogen,bounds_error=False,fill_value='extrapolate')           # Linear spline 1-D interpolation -> Operating current density - H2 consumption
        self.iHeat  = interp1d(self.CellCurrDensity,FC_Heat_produced,bounds_error=False,fill_value='extrapolate')   # Linear spline 1-D interpolation -> Operating current density - produced heat            
        self.iwater = interp1d(self.CellCurrDensity,water,bounds_error=False,fill_value='extrapolate')              # Linear spline 1-D interpolation -> Operating current density - produced water            

    ####### Operational period
    self.state = parameters["state"]                           #on or off
    self.operational_period = parameters["operational_period"]
    initial_day, final_day = self.operational_period.split(',')    #extract inital and final operational days
    initial_day = pd.to_datetime(initial_day, format = '%d-%m')
    final_day = pd.to_datetime(final_day, format = '%d-%m')
    year = initial_day.year
    operational_state = [] 
    
    for day in pd.date_range(start = pd.Timestamp(year=year,month=1,day=1), end = pd.Timestamp(year=year+1,month=1, day=1)):
        if self.state == "on":                 
            value = 0                                         #initialization
            if day >= initial_day and day <= final_day:
                value = 1                                      #update if turned on 
            operational_state.append((day, value))
        elif self.state == "off":
             value = 1                                         #initialization
             if day >= initial_day and day <= final_day:
                 value = 0                                    #update if turned off
             operational_state.append((day, value))
    operational_state = pd.DataFrame(operational_state, columns=['Day', 'State'])
    operational_state.set_index('Day', inplace=True)
    frequency =  f'{self.timestep}min'
    operational_state_freq = operational_state.resample(frequency).ffill().iloc[:-1,:]  #resample dataframe to simulation timestep
    self.operational_state = np.tile(np.array(operational_state_freq['State']), int(self.timestep_number*self.timestep/c.MINUTES_YEAR)) #repeat for simulation years

    def plot_polarizationpts(self):
        plt.figure(dpi=1000)
        plt.plot(self.CellCurrDensity,self.Voltage,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        plt.plot(self.x,self.iV1(self.x),label='linear', linestyle='--') 
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlabel('Cell Current Density [A cm$^{-2}$]')
        plt.ylabel('Stack Voltage [V]')
        plt.title('SOFC Polarization Curve' )
        plt.show()  
          
        'Polarization Curve'
         
        plt.figure(dpi=600, figsize=(9,5))
        plt.plot(self.CellCurrDensity,self.Voltage/self.nc,label='data', color='b',marker='.', linestyle='None', mec='r', markersize=7, markerfacecolor='white', zorder=0)
        plt.plot(self.CellCurrDensity,self.Ecell,label='Open Circuit Potential')
        plt.plot(self.CellCurrDensity,self.DeltaV_ohm,label='Ohmic Losses')
        plt.plot(self.CellCurrDensity,self.DeltaV_con,label='Concentration Losses')
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlabel('Cell Current Density [A cm$^{-2}$]')
        plt.ylabel('Cell Voltage [V]')
        plt.title('SOFC Polarization Curve (i-V)' )

    def plot_stackperformance(self):
        'Experimental values'
        Corrente = [3.4,20.0,25.5,30.6,40.1,55.2,65.3,80.4,90.4,100.0]                   # [A]
        VoltaggioStack = [89.0,78.1,75.7,73.7,70.1,64.8,61.3,56.3,53.0,49.9]             # [V]
        Potenza = [302.4,1563.0,1934.4,2253.5,2813.4,3576.9,4005.2,4527.0,4793.8,4990.0] # [W]
        x=np.linspace(3.4,100,100)                                                       # [A]
            
        interpolazione1=interp1d(Corrente,VoltaggioStack,bounds_error=None,kind='cubic',fill_value='extrapolate')
        interpolazione2=interp1d(Corrente,Potenza, bounds_error=None, kind='cubic', fill_value='extrapolate')
            
        fig,ax = plt.subplots(dpi=600,figsize=(9,5))
        ax2 =ax.twinx()
        ax.plot(self.Current,self.Voltage,label='V$_\mathregular{stack}$ Model')
        ax.plot(Corrente,VoltaggioStack,linestyle='None',marker='.',mec='r',markersize=10)
        ax.plot(x,interpolazione1(x),label='V data',marker='.', markersize=2)
        ax.axvline(x=100,linestyle='--',color='k')
        ax2.axhline(y=4953,linestyle='--',color='k')
        ax.grid()
        ax2.plot(self.Current,self.P*1000,label='P$_\mathregular{stack}$ Model',color='tab:red')
        ax2.plot(Corrente,Potenza,linestyle='None', marker='.',mec='b',markersize=10)
        ax2.plot(x,interpolazione2(x),label='P data',color='tab:orange',marker='.', markersize=2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=4)
        ax.set_ylim(0,None)
        ax.set_xlabel('Current [A]')
        ax.set_ylabel('Stack Tension [V]')
        ax2.set_ylabel('P$_{el}$ [W]')       
        plt.title('SOFC Stack Performance (I-V)')

    def use(self, step: int, required_power: float, available_hyd: float):
        super().use(step, required_power, available_hyd)

        available_hydrogen = self.available_hydrogen
        state = self.state

        # PowerOutput [kW] - Electric Power required from the fuel cell
        if (abs(required_power) <= self.Npower) or (available_hydrogen/self.max_h2_module < 1):      # if required power or available hydrogen in system are lower than nominal fuel cell parameters
            if abs(required_power) >= self.MinOutputPower: 
                hyd,power,FC_Heat,etaFC,water = fuel_cell.use1(self,step,p,available_hydrogen)
                if abs(hyd) > 0:
                    self.n_modules_used[step] = 1
                else:
                    self.n_modules_used[step] = 0
                        
                self.EFF[step]   = etaFC
            else:
                p = self.MinOutputPower
                hyd,power,FC_Heat,etaFC,water = fuel_cell.use1(self,step,-p,available_hydrogen)
                if abs(hyd) > 0:
                    self.n_modules_used[step] = 1
                else:
                    self.n_modules_used[step] = 0
                        
                        
                self.EFF[step]   = etaFC 

        elif abs(required_power) > self.MaxPowerStack and available_hydrogen > self.max_h2_stack:    # if required power and available hydrogen are compatible with the entire stack full-load operations
            hyd,power,FC_Heat,etaFC,water = np.array(fuel_cell.use1(self,step,-self.Npower,available_hydrogen))
            
            hyd     = hyd*self.n_modules
            power   = power*self.n_modules
            FC_Heat = FC_Heat*self.n_modules
            water   = water*self.n_modules
            
            self.n_modules_used[step]   = self.n_modules
            self.EFF[step]              = etaFC

        elif abs(required_power) > self.Npower:                                                        # if Electric Power required is higher than nominal one, i.e., more modules can be used
            required_full_modules   = min(self.n_modules,int(abs(required_power)/self.Npower))                 # number of required modules working at full load
            full_modules            = min(required_full_modules,int(available_hydrogen/self.max_h2_module))  # number of modules operating full load based on the amount of hydrogen available   
            hyd,power,FC_Heat,etaFC_full,water = np.array(fuel_cell.use1(self,step,-self.Npower,available_hydrogen))
            
            hyd_full        = hyd*full_modules
            power_full      = power*full_modules
            FC_Heat_full    = FC_Heat*full_modules
            water_full      = water*full_modules
            
            residual_power      = abs(required_power) - power_full 
            residual_hydrogen   = available_hydrogen - abs(hyd_full)
            if residual_power >= self.MinOutputPower:
                hyd_singlemodule,power_singlemodule,FC_Heat_singlemodule,etaFC_singlemodule,water_singlemodule = fuel_cell.use1(self,step,-residual_power,residual_hydrogen)
                if hyd_singlemodule != 0:   # single module operating in partial load 
                    self.n_modules_used[step] = full_modules + 1
                    self.EFF[step] = ((full_modules*etaFC_full) + (etaFC_singlemodule))/self.n_modules_used[step]  # weighted average 
                    self.EFF_last_module[step] = etaFC_singlemodule
                else: 
                    self.n_modules_used[step] = full_modules
                    self.EFF[step] = etaFC_full
            else:
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
        p_required = abs(required_power )# [kW] 
        FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea   # [A/cm^2] current density value at which the fuel cell is working
        Current = FC_CellCurrDensity*self.FC_CellArea               # [A] Fuel Cell module operating current 
        
        # Checking if resulting current density is high enough for the fuel cell to start, otherwise hydrogen used = 0
        if FC_CellCurrDensity < self.FC_MinCurrDens:      # condition for operability set for current density 
            etaFC               = 0     # [-] fuel cell efficiency
            hyd                 = 0     # [kg/s] hydrogen used in the considered timestep
            Current             = 0     # [A] Operational Current
            p_required          = 0     # [kW] required energy - when timestep is kept at 1 h kW = kW
            FC_Heat             = 0     # [kW] thermal energy produced
            FC_CellCurrDensity  = 0     # [A/cm2] fuel cell current density
            water               = 0     # [Sm3/s] water production
        else: 
            FC_Vstack= self.iV1(FC_CellCurrDensity)            # [V] Stack operating voltage
            
            etaFC       = float(self.iEta(FC_CellCurrDensity))         # [-] FC efficiency                
            hyd         = float(self.ihyd(FC_CellCurrDensity))         # [kg/s] hydrogen consumed  
            FC_Heat     = float(self.iHeat(FC_CellCurrDensity))        # [kW] thermal energy produced
            water       = float(self.iwater(FC_CellCurrDensity))       # [m^3/s] water production 

            if hyd > available_hydrogen:     # if not enough hydrogen is available in the system to meet demand (H tank is nearly empty)
               # defining the electric load that can be covered with the hydrogen available 
                hyd,p_required,FC_Heat,etaFC,water = fuel_cell.h2power(self,available_hydrogen)

        return (-hyd,p_required,FC_Heat,etaFC,water)  # return hydrogen absorbed [kg/s] electricity required [kW] and heat as a co-product [kW]
