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

class FuelCellState(Enum):
    """Defines all the states of a fuel cell.
    ON - In a specified time frame (FuelCellParameters#operational_period) the fuel cell is allowed to run. On other days the fuel cell is shut off.
    OFF - In a specified time frame (FuelCellParameters#operational_period) the fuel cell is shut off. On other days the fuel cell is allowed to run.
    """
    ON = 0
    OFF = 1

@dataclass
class FuelCellParameters:
    nominal_power: float
    """The nominal power of one single module.
    Value is in kW
    """

    module_number: int
    """Number of modules in the stack
    """

    minimum_power_per_module: float = 0
    """A float in [0;1].
    Specifies the minimum load (relative to max load) the modules of the fuel cell must run at.
    Default: 0
    """

    priority: int
    """The priority of this tech.
    """

    calcluate_ageing: bool = False
    """Whether to calculate the ageing of the fuel cell.
    Default: False

    NOTE: Not yet implemented by MESSpy. Therefore setting this to True does nothing
    """

    operational_period: str
    """The period in which the fuel cell is running in the course of any year.
    Format: dd-mm,dd-mm.
    Example: 01-03,31-10 => fuel cell runs from March 1 to October 31. The other days in the year, the fuel cell is shut off.
    """

    state: FuelCellState
    """Defines whether the fuel cell is on or off in a given self#operational_period.
    For more information see FuelCellState
    """

    timestep: int = 60
    """The length of one timestep in min. 
    Must be larger than 0
    Default: One hour
    NOTE: This value may be overridden by a main-script.
    """

class FuelCell(abc.ABC):
    @abc.abstractmethod
    def __init__(self, parameters: FuelCellParameters, timestep_number: int):
        """Create a fuel cell object.
        Absorbs hydrogen and produces electricity, heat and water.
        See self#use

        Parameters
        ----------
        parameters: FuelCellParameters
            All the parameters the fuel cell needs to be instantiated

        timestep_number: int
            The number of timesteps the fuel cell should go through

        timestep: boolean = False
            TODO
        """
        self._set_consts()
        self.parameters = parameters
        self.timestep_number = timestep_number
        self.timestep = timestep

        self.MaxPowerStack = self.parameters.nominal_power * self.parameters.module_number
        self.MinOutputPower = self.parameters.nominal_power * self.parameters.minimum_power_per_module

        if __name__ == "__main__":
            # if code is being executed from fuel_cell.py script
            self.timestep = timestep
            self.timestep_number = timestep_number
        else:
            self.timestep = c.timestep
            self.timestep_number = c.timestep_number
            
        self.timesteps_year =  self.min_year/self.timestep
        self.timesteps_week =  self.min_week/self.timestep

    @abc.abstractmethod
    def plot_polarizationpts(self):
        pass

    @abc.abstractmethod
    def plot_stackperformance(self):
        pass

    @abs.abstractmethod
    def use(self, step: int, required_power: float, available_hyd: float):
        """One step of the simulation
        Hydrogen is absorbed and produces electricity.
        Reaction: H2 ---> 2H+ + 2e-

        Parameters
        ----------
        step: int
            The step to be simulated (an index to the operational_period)

        required_power: float
            The power in kW the system requires at the given step.
            Value will be used as abs, so it does not matter if the input is positive or negative.

        available_hyd: float
            The hydrogen in kg that can be supplied by the system at the current step

        Returns
        -------
        tuple[float, float, float, float]
            0: hydrogen consumption [kg/s]
            1: electricity [kW] 
            2: heat [kW]
            3: water [Sm^3]
        """
        self.available_hydrogen = available_hyd/(self.timestep*60) # [kg] to [kg/s] conversion for available hydrogen at the considered step
        self.state = self.operational_state[step]
        if self.state == 0:
            required_power = 0   # # fuel cell turned off as for planned operation schedule, required power output forced to zero

        return (0,0,0)

    def _set_consts(self):
        self.min_year   = c.MINUTES_YEAR                    # [min/year]    number of minutes in one year
        self.min_week   = c.MINUTES_WEEK                    # [min/week]    number of minutes in one week
        self.min_month  = c.MINUTES_MONTH                   # [min/month]   number of minutes in one month        
        self.rhoStdh2        = c.H2SDENSITY         # [kg/Sm3]    PropsSI('D', 'T', 288.15, 'P', 101325, 'H2') H2  density @ T = 15°C p = 101325 Pa
        self.rhoStdh2o       = c.H2OSDENSITY        # [kg/m3]     H2O density @ T = 15°C p = 101325 Pa
        self.Runiv           = c.R_UNIVERSAL        # [J/(mol*K)]
        self.Rh2             = c.R_H2               # [J/(kg*K)] 
        self.FaradayConst    = c.FARADAY            # [C/mol]     Faraday constant
        self.deltaG0         = c.GIBBS              # [kJ/mol]    Gibbs free energy @ T = 25°C p = 101325 Pa
        self.GammaPerfectGas = c.GAMMA              # [-]         Gamma = cp/cv  
        self.LHVh2           = c.LHVH2              # [MJ/kg]     H2 LHV
        self.HHVh2           = c.HHVH2              # [MJ/kg]     H2 HHV
        self.HHVh2Mol        = c.HHVH2MOL           # [kJ/mol]    H2 HHV molar
        self.cpH2            = c.CP_H2              # [kJ/(kgK)]  Hydrogen specific heat @ T = 25°C, P = 101325 Pa
        self.cpH2O           = c.CP_WATER           # [kJ/(kgK)]  Water specific heat
        self.cpAir           = c.CP_AIR             # [kJ/(kgK)]  Air specific heat @ T = 25°C, P = 101325 Pa
        self.h2oMolMass      = c.H2OMOLMASS         # [kg/mol]    Water molar mass
        self.H2MolMass       = c.H2MOLMASS          # [kg/mol]    Hydrogen molar mass
        self.AirMolMass      = c.AIRMOLMASS         # [g/mol]      Air molar mass
        self.H2MolStdEntropy = c.H2MOL_S_E          # [J/K*mol]    Specific molar entropy (gaseous phase)
        self.O2MolStdEntropy    = c.O2MOL_S_E       # [J/K*mol]    Specific molar entropy
        self.H20MolStdEntropy   = c.H2OMOL_S_E      # [J/K*mol]    Specific molar entropy
        self.SteamSH            = c.H1_STEAM800     # [kJ/kg]     Steam mass specific enthalpy @ T = 800°C, P = 116000 Pa

        # Math costants
        self.eNepero      = c.NEPERO                # [-]         Euler's number
        # Ambient conditions 
        self.AmbTemp      = c.AMBTEMP               # [K]         Standard ambient temperature - 15 °C

    def h2power(self,hyd):
        """
        Inverse function that computes fuel cell efficiency, electric power output and thermal power output based on hydrogen consumption

        Parameters
        ----------
        hyd : float exploitable hydrogen to produce electricity in the timestep [kg/s]

        Returns
        -------
        hyd: float exploited hydrogen in the timestep [kg/s] (same as input 'h2')
        p_required : float electricity produced in the timestep [kW]
        FC_Heat : Heat produced by Fuel cell operation in the time step [kW]
        etaFC : module efficiency [-]

        """
        if 0 <= hyd <= self.max_h2_module:     # if lower than maximum consumption capacity
            
            if self.ageing:
                p_required  = self.hydP(hyd)
                FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea   # [A/cm^2] current density value at which the fuel cell is working  
                if FC_CellCurrDensity < self.FC_MinCurrDens or p_required < self.MinOutputPower:
                    p_required    = 0       # [kW]    required energy - when timestep is kept at 1 h kW = kW
                hyd,p_required,FC_Heat,etaFC,water = fuel_cell.ageing(self,step,p_required)
            
            else:
                p_required         = self.h2P(hyd)                          # [kW] coverable electric power
                FC_CellCurrDensity = self.PI(p_required)/self.FC_CellArea   # [A/cm^2] current density value at which the fuel cell is working 
    
                if FC_CellCurrDensity < self.FC_MinCurrDens or p_required < self.MinOutputPower:      # condition for operability set for current density 
                    etaFC         = 0       # [-]      fuel cell efficiency
                    hyd           = 0       # [kg/s]     hydrogen used in the considered timestep
                    Current       = 0       # [A]      Operational Current
                    p_required    = 0       # [kW]    required energy - when timestep is kept at 1 h kW = kW
                    FC_Heat       = 0       # [kW]    thermal energy used
                    FC_CellCurrDensity = 0  # [A/cm^2] current density
                    water              = 0  # [m^3/s] water production
                else: 
                    etaFC       = float(self.etaFuelCell(hyd))  # [-]   FC efficiency
                    FC_Heat     = float(self.FC_Heat(hyd))      # [kW] FC produced heat
                    water       = float(self.water(hyd))        # [m^3/s] water production
            
        return(hyd,p_required,FC_Heat,etaFC,water)

    def tech_cost(self,tech_cost):
        """
        Parameters
        ----------
        tech_cost : dict
            'cost per unit': float [€/kW]
            'OeM': float, percentage on initial investment [%]
            'refud': dict
                'rate': float, percentage of initial investment which will be rimbursed [%]
                'years': int, years for reimbursment
            'replacement': dict
                'rate': float, replacement cost as a percentage of the initial investment [%]
                'years': int, after how many years it will be replaced

        Returns
        -------
        self.cost: dict
            'total cost': float [€]
            'OeM': float, percentage on initial investment [%]
            'refud': dict
                'rate': float, percentage of initial investment which will be rimbursed [%]
                'years': int, years for reimbursment
            'replacement': dict
                'rate': float, replacement cost as a percentage of the initial investment [%]
                'years': int, after how many years it will be replaced
        """
        tech_cost = {key: value for key, value in tech_cost.items()}

        size = self.Npower * self.n_modules
        
        if tech_cost['cost per unit'] == 'default price correlation':
            C = (self.n_modules*self.nc*self.FC_CellArea*(2.96*self.FC_OperatingTemp-1907))/(10**(4))
        else:
            C = size * tech_cost['cost per unit']

        tech_cost['total cost'] = tech_cost.pop('cost per unit')
        tech_cost['total cost'] = C
        tech_cost['OeM'] = tech_cost['OeM'] *C /100 # €


        self.cost = tech_cost  

    def ageing(self,step,power):
        """
        Computes the ageing effects on a fuel cell, adjusting performance by modeling voltage increases 
        due to operational time and temperature changes, which in turn impact hydrogen production efficiency.
    
        Parameters:
            step (int): Current simulation step indicating operational time
            power (float): Electrical power output to be provided by the fuel cell [kW]
    
        Returns:
            hyd_consumed (float): Adjusted hydrogen production rate in kg/s, accounting for ageing effects
            power (float): The output power provided based on funciton calculations, reflecting the operational status of the fuel cell
            P_th (float): Thermal power output in kW, adjusted for ageing
            eta (float): Efficiency of the fuel cell, adjusted for the impact of ageing
            water (float): Amount of water produced in standard cubic meters per second, adjusted for ageing
        
        This function updates the fuel cell's polarization curve to reflect degradation and utilizes this curve 
        to determine new operational parameters, including hydrogen production rate. It also performs degradation 
        rate calculations using polynomial fitting on log-transformed data and updates internal tracking of 
        module efficiency and polarization curve history.
        """
        def deg_rate(self,φ,plot=True):
            """
            Estimates the degradation rate of a FuelCell based on the load profile value (φ) using a polynomial fit on log-transformed data.
            Optionally plots the data with the fitting curve and displays the goodness of fit (R²).

            φ: Load profile characteristic value(s) for which degradation rate is calculated.
            plot: If True, plots the fitting results.

            Returns the estimated degradation rate using the exponential of the fitted polynomial.
            """
            
            # experimental dataset linking characteristic load profile value and degradation rate
            # ref. https://doi.org/10.1016/j.ijhydene.2017.02.146
            dataset = { 
                        'φ': np.array([1,1,1,1,1,3,4.3,5,7,7.5,9.2,9]),                 # [-] load profile characteristic value
                        'φ°':np.array([4,1,2,11,6,50,50,75,200,300,260,400])            # [μV/h]  voltage decrease - degradation rate
                        }
            
            # Aggregate the data by averaging the φ° values for each unique φ
            unique_phi      = np.unique(dataset['φ'])
            average_phi_dot = np.array([np.mean(dataset['φ°'][dataset['φ'] == val]) for val in unique_phi])
            
            # Log transformation of the output to ensure positivity
            average_phi_dot_log = np.log(average_phi_dot)  # applying log function to dataset
            
            degree  = 3  # choosing the degree for the interpolating polynomial function
            
            coefficients = np.polyfit(unique_phi,average_phi_dot_log,degree)  # fitting log of data series
            polynomial = np.poly1d(coefficients) # interpolating function creatio
            
            def exp_poly(x):
                """
                Applies the exponential function to the polynomial model's output to transform it back to the original scale.
                x: Input value(s) for which to calculate the degradation rate.
                Returns the exponential of the polynomial model's output, ensuring all values are non-negative.
                """
                return np.exp(polynomial(x))  # expanding result to return o the original scale
            
            φ_d     = exp_poly(φ)               # [μV/h] interpolated and transformed value
            φ_dot   = φ_d*(1e-6)/60             # [V/min] measure units conversion 
            
            if plot == True and step == 0:
                # test
                phi_new = np.linspace(min(unique_phi), max(unique_phi), 1000)
                phi_dot_pred = exp_poly(phi_new)
                
                # Calculate R²
                residuals = average_phi_dot_log - polynomial(unique_phi)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((average_phi_dot_log - np.mean(average_phi_dot_log))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Plotting
                plt.scatter(unique_phi, average_phi_dot, label='Averaged Data')
                plt.plot(phi_new, phi_dot_pred, color='red', label='Approximating Polynomial (Exp Transformed)')
                plt.xlabel('φ [-]')
                plt.ylabel('φ° [μV/h]')
                plt.title(f'Log-Transformed Polynomial Interpolation (R²={r_squared:.4f})')
                plt.legend()
                plt.show()

            return φ_dot
        
        'Parameters definition'
        # k1 and k2 are the constants to be used in the weight functions for voltage and current characteristic values.
        # It is recommended to use values of k1 >= 25 and k2 >= 5 for model accuracy.

        # Current parameter
        k_1 = 25      # [-] wight function constant
        # Voltage
        k_2 = 6.5     # [-] wight function constant
        
        if power <= 0:  # fuel cell not working
            iop_id      = 0
            v_op        = 0
            operation   = False
        else:           # if the fuel cell has been activated at current step
            self.stack['Activation[-]'][step] = 1
            
            'Ideal behaviour' 
            iop_id    = self.Pi(power)      # [A/cm^2] module operating ideal current density based on system power output
            Iop_id    = self.PI(power)      # [A] module operating ideal current based on system power output
            Vop_id    = self.PV(power)      # [V] module operating ideal voltage based on system power output 
            H2op_id   = self.Ihyd(Iop_id)   # [kg/s] module operating ideal hydrogen consumption based on system power output 
            Pthop_id  = self.IHeat(Iop_id)  # [kW] module operating ideal by-produced heat based on system power output 
            Etaop_id  = self.IEta(Iop_id)   # [-] module operating ideal efficiency based on system power output 
            H2Oop_id  = self.Iwater(Iop_id) # [Sm3/s] module operating water production based on system power output
            operation = True
        
        ageing_factor_rated = max(self.polarization_curve_ageing)/max(self.Voltage) # [-] ageing factor for functioning at rated power
        self.stack['Conversion_ratio_rated[kWh/kg]'][step]  = self.Γ*ageing_factor_rated
        # link between current and module voltage: polarization curve
        if operation == True:  # if fuel cell is working in current step
            IV_new   = interp1d(self.Current,self.polarization_curve_ageing) # Linear spline 1-D interpolation - updating I-V function for ageing effect
            V_op                = IV_new(Iop_id)       # [V] operational voltage accounting for ageing effect 
            v_op                = V_op/self.nc      # [V] operational cell voltage accountig for ageing
            ageing_factor_op    = V_op/Vop_id     # [-] ageing factor expressed as the ratio between operational and ideal voltage for the considered current. Numerator decreases over time
            hyd_consumption     = H2op_id/ageing_factor_op                      # [kg/s] hydrogen consumption in operative conditions accounting for ageing effects
            P_th                = Pthop_id/ageing_factor_op             # [kW] thermal power output
            eta                 = Etaop_id*ageing_factor_op             # [-] module operating efficiency corrected with ageing factor
            water               = H2Oop_id//ageing_factor_op            # [Sm^3/s] water production
            self.stack['Conversion_ratio_op[kWh/kg]'][step]     = self.Γ*ageing_factor_op                
            hyd_cons            = self.hydcons/ageing_factor_op
            self.hydP           = interp1d(hyd_cons,self.P)     # updating interpolation function used in h2power function
        else:
            hyd_consumption     = 0
            power               = 0
            P_th                = 0
            eta                 = 0
            water               = 0
        
        self.stack['i_op[A]'][step]     = iop_id        # [A/cm^2]       operating current density
        self.stack['v_op[V]'][step]     = v_op          # [V]   cell operating voltage accounting for ageing         
        
        'Weekly ageing phenomena computation'
        if step % self.timesteps_week == 0 and step != 0: # updating the polarization curve every week 
            start_index = int(max(0,step-self.timesteps_week))        
            
            # considering operational current and voltage values for the period under consideration
            load_i = self.stack['i_op[A]'][start_index:step]
            load_V = self.stack['v_op[V]'][start_index:step]
        
            # operating time counter
            operation_time  = sum(self.stack['Activation[-]'][start_index:step]) # number of timesteps the fuel cell has been operating
            
            if operation_time != 0:

                'Load profile - current density'
                # Discrete Fourier Transform - DFT
        
                DFT_i=scipy.fft.fft(load_i)     # Fast Fourier Transorm (Discretized)
                # Determine the number of points in DFT and create an index array
                N = len(DFT_i)
                n = np.arange(N)
                
                # defining sampling frequency and calculating the total time
                Fs      = 1/(60*self.timestep)    # [Hz] sampling frequency
                T       = N/Fs
                freq    = n/T # frequency bins for DFT
                
                # defining the maximum frequency to plot based on the Nyquist criterion
                f_max = min(Fs, freq[-1])
        
                DFT_i_mag=np.abs(DFT_i)/N # normalizing DFT magnitude
    
                def DFT_current():
                    plt.figure(dpi=1000)      
                    plt.plot(freq[0:int(N/2+1)],2*DFT_i_mag[0:int(N/2+1)])
                    plt.grid()
                    plt.xlim(0,f_max)
                    plt.ylabel('|FFT_i(J)| [A/cm^2]')
                    plt.xlabel('Frequency (Hz)')
                    plt.title('Single-Sided Amplitude Spectrum of J(t)')
                    plt.show() 
    
                w_curr = [k_1,1] # defining weight for current load analysis
                # Fit a polynomial to the absolute value of the DFT over frequency
                p_Cln = np.polyfit(freq, np.abs(DFT_i), 3)  # Third degree polynomial fitting
                # Integrate the product of the polynomial fit and the current weight
                prod = np.polyint(np.poly1d(p_Cln)*np.poly1d(w_curr))      
                # Evaluate the definite integral between f_max and 0 to find the current modification factor
                I = np.polyval(prod,f_max)-np.polyval(prod,0)
                self.m_curr = round(1/T*I+1,6)
    
                'Load profile - cell voltage'
                # Voltage histogram
                
                # Define bins for the histogram. These bins cover the range from slightly below the minimum voltage (v_0)
                # to slightly above the maximum voltage (vol_max), with intervals of 0.01 volts
                # capturing the distribution of voltage values
                self.bins = np.arange(self.v_0 - 0.01, self.vol_max + 0.01, 0.01)  # [V] Voltage bins
                self.v_round = np.around(load_V,3)  # [V] rounded voltage measurements
                # indices where the voltage is greater than 0, to consider only positive voltage readings
                self.v_pos = np.where(self.v_round > 0)[0]  # indices of positive voltages
    
                # voltage load profile histogram
                self.v_counts = np.histogram(self.v_round[self.v_pos], self.bins)[0]  # voltage counts in each bin
    
                # normalizing the histogram by voltage measurements to get the frequency distribution
                self.H_v = self.v_counts/len(self.v_pos)  # normalized histogram frequencies
    
                # calculating the total of the normalized histogram frequencies
                # ideally be close to 1 if all measurements are accounted for and correctly binned
                self.H_v_tot = self.H_v.sum()  # Sum of normalized frequencies
         
            
                def v_count_plot():
                    """
                    Plots a histogram of voltage counts across specified bins to visualize the distribution of operation voltages. 
                    It highlights the lower (v_L) and upper (v_U) limits of the optimal voltage range with dashed red lines. 
                    This visualization helps in assessing the frequency of voltages within and outside the optimal operating conditions.
                    """
                    fig = plt.figure(dpi=600)    
                    ax = fig.add_subplot(111)  
                    self.v_counts = ax.hist(self.v_round, self.bins, density=False, facecolor='cornflowerblue', edgecolor='black', rwidth=0.6, zorder=3)
                    ax.vlines(x=self.v_L, ymin=0, ymax=max(self.v_counts[0])+1, linewidth=1.5, color="indianred", linestyle="dashed", zorder=4)
                    ax.vlines(x=self.v_U, ymin=0, ymax=max(self.v_counts[0])+1, linewidth=1.5, color="indianred", linestyle="dashed", zorder=4)
                    ax.text(self.v_L-0.06, max(self.v_counts[0]), "Voltage$_{min}$", fontsize=8, horizontalalignment='center', zorder=5)
                    ax.text(self.v_U+0.06, max(self.v_counts[0]), "Voltage$_{max}$", fontsize=8, horizontalalignment='center', zorder=5)
                    ax.set_xlim([self.v_0-0.1, self.vol_max+0.1])
                    ax.set_ylim([0,max(self.v_counts[0])+1])
                    ax.set_xlabel('Operation voltage range [V]')
                    ax.set_ylabel('Voltage counts [-]')
                    ax.grid(True, zorder=0, alpha=0.4)
                    plt.show()
                
                # v_count_plot()
            
                def H_v_plot():
                    fig = plt.figure(dpi=600)    
                    ax = fig.add_subplot(111)
                    # Convert the histogram frequencies to percentages
                    percentage_v_counts = self.H_v * 100
                    counts, _, _ = ax.hist(self.bins[:-1], bins=self.bins, weights=percentage_v_counts, density=False, facecolor='cornflowerblue', edgecolor='black', rwidth=0.6, zorder=3)
                    ax.vlines(x=self.v_L, ymin=0, ymax=max(percentage_v_counts)+1, linewidth=1.5, color="indianred", linestyle="dashed", zorder=4)
                    ax.vlines(x=self.v_U, ymin=0, ymax=max(percentage_v_counts)+1, linewidth=1.5, color="indianred", linestyle="dashed", zorder=4)
                    ax.text(self.v_L-0.06, max(percentage_v_counts), "Voltage$_{min}$", fontsize=8, horizontalalignment='center', zorder=5)
                    ax.text(self.v_U+0.06, max(percentage_v_counts), "Voltage$_{max}$", fontsize=8, horizontalalignment='center', zorder=5)
                    ax.set_xlim([self.v_0-0.1, self.vol_max+0.1])
                    ax.set_ylim([0, max(percentage_v_counts)+1])
                    ax.set_xlabel('Operation voltage range [V]')
                    ax.set_ylabel('Voltage distribution [%]')
                    ax.grid(True, zorder=0, alpha=0.4)
                    plt.show()
    
                H_v_plot()
            
                self.bins = self.bins[1:]
                
                # defining weight for voltage load analysis w_vol
                w_vol_1 = lambda v: (k_2*(v-self.v_L))**2+1  # if v<self.v_L
                w_vol_2 = 1                                  # if self.v_L<v<self.v_U
                w_vol_3 = lambda v: (k_2*(v-self.v_U))**2+1  # if v>self.v_U
                
                bin_width   = self.bins[1]-self.bins[0] # [V] total width of every bin  in the selected interval
                bin_center  = bin_width/2               # [V] to subtract from bin value in order to obtain the average value of the bin among the interval extremes
                
                # voltage value (summation method)
                m = 0
                for index in range(len(self.bins)):
                    
                    if self.bins[index] < self.v_L:
                        val = self.H_v[index]*w_vol_1(self.bins[index]-bin_center)
                    elif self.bins[index] >= self.v_L and self.bins[index] <= self.v_U:
                        val = self.H_v[index]*w_vol_2
                    elif self.bins[index]>self.v_U:
                        val = self.H_v[index]*w_vol_3(self.bins[index]-bin_center)
                    m+=val
                    
                self.m_vol=m
                        
                'φ parameter calculation'
                ## characteristic value of load
                self.φ = self.m_curr*self.m_vol
                
                V_deg       = deg_rate(self,self.φ)                 # [V/min]  voltage decrease in the considered period - degradation rate
                V_operation = V_deg*(operation_time*self.timestep)  # [V] voltage loss for the single fc cell due to operational conditions in the considered period
                
                # updating polarization curve
                self.polarization_curve_ageing -= V_operation*self.nc  # [V] self.Voltage represents the design polarization curve
    
                # limit on degradation for single cell voltage reached
                if max(self.polarization_curve_ageing-V_operation*self.nc)/self.nc > self.CellVoltage_limit:
                    print('Electorlyzer module voltage exceeds safe limits due to ageing. Module must be replaced')
            
        self.stack['hydrogen_consumption[kg/s]'][step]           = hyd_consumption  # [kg/s]    hydrogen consumption in the timestep
        
        if step % self.timesteps_year == 0:
            self.stack['Pol_curve_history'].append(self.polarization_curve_ageing.copy())
            self.stack['Module_efficiency[-]'].append(self.eta_module*(self.polarization_curve_ageing/self.Voltage))         # [kg/MWh] ideal converison factor
            print(f'Year {int(step/self.timesteps_year)}')
        
        return hyd_consumption,power,P_th,eta,water
