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

class fuel_cell:

if __name__ == "__main__":
    
    """
    Functional test
    """

    inp_test = {'Npower': 1000,
                "number of modules": 1,
                'stack model':'PEM General',
                'electric efficiency':  0.45,
 				'thermal efficiency':  0.35,
                'ageing': False,
     			'min power module' : 0.2,
                'operational_period': "01-01,31-12",
                'state': "on"}
         
    
    if inp_test['ageing'] == False: 
        sim_steps   = 8760                           # [-] number of steps to be considered for the simulation - usually a time horizon of 1 year minimum is considered
        timestep    = 60                               # [min] selected timestep for the simulation
        time        = np.arange(sim_steps)
        
        fc = fuel_cell(inp_test,sim_steps,timestep=timestep)         # creating fuel cell object
        # fc.plot_polarizationpts()                  # cell polarization curve
        
        available_hydrogen = 1000                   # [kg] hydrogen available in the storage system
    
        if fc.model == 'PEM General' and fc.Npower == 13.6:
            fc.plot_stackperformancePEM()                   # PEMFC stack performance curve
        if fc.model == 'SOFC' and fc.Npower == 5:
            fc.plot_stackperformanceSOFC()                  # SOFC stack performance curve
    
        'Test 1 - Tailored ascending power input'
    
        flow  = - np.linspace(fc.Npower*0.01,fc.Npower*6,sim_steps)  # [kW] power demand - ascending series
        flow1 = - np.linspace(fc.Npower*0.01,fc.Npower,sim_steps)    # [kW] power demand - ascending series
    
        hyd_used = np.zeros(sim_steps)      # [kg] hydrogen used by fuel cell
        P_el     = np.zeros(sim_steps)      # [kW] electricity produced
        P_th     = np.zeros(sim_steps)      # [kW] produced heat
        eta      = np.zeros(sim_steps)      # [-]  efficiency  
        water    = np.zeros(sim_steps)      # [Sm3/s]  produced water  
        
        for step in range(len(flow1)):
            hyd_used[step],P_el[step],P_th[step],eta[step],water[step] = fc.use(step,flow1[step],available_hydrogen)
            # available_hydrogen += hyd_used[step]*60*timestep  # [kg] updating available hydrogen
            
        # fc.EFF[fc.EFF == 0] = math.nan      # activate to avoid representation o '0' values when fuel cell is turned off
                                  
        
        fig=plt.figure(figsize=(8,8),dpi=1000)
        fig.suptitle("{} ({} kW) performance".format(inp_test['stack model'],round(fc.Npower,1)))
        
        PI=fig.add_subplot(211)
        PI.plot(-flow1,P_th,label="Thermal Power") 
        if inp_test['stack model'] != 'simple':
            PI.axvline(x=fc.P[0],linestyle=':',color='tab:red',label= 'Lower Functioning Boundary', zorder=3, linewidth = 2)  
            PI.axvline(x=fc.MinOutputPower,linestyle=':',label= 'Minimum Output Power', zorder=3, linewidth = 2)
        PI.set_title("Heat vs Electric Power")
        PI.grid(alpha=0.3, zorder=-1)
        PI.set_xlabel("Power Demand [kW]")
        PI.set_ylabel("Thermal Output [kW]")
        PI.legend(fontsize=15)
        
        if inp_test['stack model'] != 'simple':
            ETA=fig.add_subplot(212)
            ETA.scatter(-flow1,fc.EFF,label="Efficiency",color="green",edgecolors='k')
            ETA.axvline(x=fc.MinOutputPower,color='tab:blue',linestyle=':',label= 'Minimum Output Power', zorder=3, linewidth = 2)   
            ETA.axvline(x=fc.P[0],linestyle=':',color='tab:red',label= 'Lower Functioning Boundary', zorder=3, linewidth = 2)   
            ETA.set_title("Efficiency vs Power")
            ETA.grid(alpha=0.3, zorder=-1)
            ETA.set_xlabel("Power Demand [kW]")
            ETA.set_ylabel("Efficiency [-]")
            ETA.legend(fontsize=15)
        
        plt.tight_layout()
        plt.show()
        
        
        fig=plt.figure(figsize=(8,8),dpi=1000)
        fig.suptitle("{} ({} kW) performance".format(inp_test['stack model'],round(fc.Npower,1)))
        
        PI=fig.add_subplot(211)
        PI.plot(-flow1,-hyd_used)
        if inp_test['stack model'] != 'simple':
            PI.axvline(x=fc.P[0],linestyle=':',color='tab:red',label= 'Lower Functioning Boundary', zorder=3, linewidth = 2) 
            PI.axvline(x=fc.MinOutputPower,linestyle=':',label= 'Minimum Output Power', zorder=3, linewidth = 2) 
        PI.set_title("H$_{2}$ Consumption vs Power")
        PI.grid(alpha=0.3, zorder=-1)
        PI.set_xlabel("Power Output [kW]")
        PI.set_ylabel("Hydrogen consumption [kg/s]")
        # PI.legend(fontsize=15)
        
        if inp_test['stack model'] != 'simple':
            ETA=fig.add_subplot(212)
            ETA.scatter(-flow1,fc.EFF,label="Efficiency",color="green",edgecolors='k',zorder =3)
            ETA.axvline(x=fc.MinOutputPower,color='tab:blue',linestyle=':',label= 'Minimum Output Power', zorder=3, linewidth = 2)   
            ETA.axvline(x=fc.P[0],linestyle=':',color='tab:red',label= 'Lower Functioning Boundary', zorder=3, linewidth = 2)
            ETA.set_title("Efficiency vs Power")
            ETA.grid()
            ETA.set_xlabel("Power Output [kW]")
            ETA.set_ylabel("Efficiency [-]")
            ETA.legend(fontsize=15)
        
        plt.tight_layout()
        plt.show()
        
        if inp_test['stack model'] != 'simple':
            fig, ax = plt.subplots(dpi=600)
            ax.scatter(-flow1,fc.EFF, edgecolors='k', zorder = 3)
            ax.set_title("Fuel Cell Module Efficiency")
            textstr = '\n'.join((
                r'$CellArea=%.1f$ $cm^{2}$' % (fc.FC_CellArea,),
                r'$P_{nom}= %.1f$ kW' % (fc.Npower,),
                r'$i_{max}= %.1f$ A $cm^{-2}$' % (fc.FC_MaxCurrDens,),
                r'$n_{cell}= %.0f$' % (fc.nc,)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(fc.Npower/2,0.2,textstr,fontsize=10,va='bottom',backgroundcolor='none', bbox=props)
            ax.grid(alpha=0.3, zorder=-1)
            ax.set_ylim(0,None)
            ax.set_xlabel('Power Output [kW]')
            ax.set_ylabel('$\\eta$') 
            
            plt.figure(dpi=1000)
            plt.plot(-flow1, fc.EFF)
            plt.grid(alpha=0.3,zorder=-1)
            plt.xlabel('Power Output [kW]')
            plt.ylabel('$\\eta$ - Efficiency [-]')
           
        for step in range(len(flow)):
            hyd_used[step],P_el[step],P_th[step],eta[step],water[step] = fc.use(step,flow[step],available_hydrogen)
         
        if inp_test['stack model'] != 'simple':
            fig, ax = plt.subplots(dpi=600)
            ax.plot(-flow,fc.n_modules_used,color='tab:green',zorder=3)
            ax.set_xlabel('Required Power [kW]')
            ax.set_ylabel('Active modules [-]')
            ax.grid(alpha=0.3, zorder=-1)
            ax.set_title('Fuel Cell Stack - Nr of working modules')
          
        
        'Test 2 - Random power demand'
    
        fd   = -np.random.uniform(0.08*fc.Npower,5.2*fc.Npower,sim_steps)   # [kW] power required from the Fuel Cell - random values
        
        for step in range(len(fd)):
            hyd_used[step],P_el[step],P_th[step],eta[step],water[step] = fc.use(step,fd[step],available_hydrogen)
        
        if inp_test['stack model'] != 'simple':
                         
                                   
            fig, ax = plt.subplots(dpi=1000)
            ax2 = ax.twinx() 
            ax.bar(np.arange(sim_steps)-0.2,fc.EFF,width=0.35,zorder=3,edgecolor='k',label='$1^{st}$ module efficiency', alpha =0.8)
            ax.bar(np.arange(sim_steps)+0.,fc.EFF_last_module,width=0.35,zorder=3, edgecolor = 'k',align='edge',label='Last module efficiency',alpha =0.8)
            ax2.scatter(np.arange(sim_steps),-fd,color ='limegreen',s=25,edgecolors='k',label='Required Power')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='lower center',bbox_to_anchor=(0.5, 1.08), ncol =3, fontsize ='small')
            ax.set_xlabel('Time [step]')
            ax.set_ylabel('Efficiency [-]')
            ax2.set_ylabel('Power Output [kW]')
            ax.grid(alpha=0.3, zorder=-1)
            ax.set_title('Fuel Cell Stack functioning behaviour')
                        
            num = 24   # number of hours to be represented in the plot below
            
            fig, ax = plt.subplots(dpi=1000)
            ax.bar(np.arange(num)-0.2,P_el[:num],width=0.35,zorder=3,color='lightseagreen',edgecolor='k',label='Electric output', alpha =0.8)
            ax.bar(np.arange(num)+0.,P_th[:num],width=0.35,zorder=3,color='indianred',edgecolor='k',align='edge',label='Thermal output',alpha =0.8)
            ax.legend(loc='lower center',bbox_to_anchor=(0.5, 1.08), ncol =3, fontsize ='small')
            ax.set_xlabel('Time [step]')
            ax.set_ylabel('Power [kW]')
            ax2.set_ylabel('Power Output [kW]')
            ax.grid(alpha=0.3, zorder=-1)
            ax.set_title('Fuel Cell Stack functioning behaviour')
        
#%%
    elif inp_test['ageing'] == True:
        inp_test['number of modules'] = 1
        
        sim_steps   = 8760*4                          # [-] number of steps to be considered for the simulation - usually a time horizon of 1 year minimum is considered
        timestep    = 60                               # [min] selected timestep for the simulation
        time        = np.arange(sim_steps)
        
        fc = fuel_cell(inp_test,sim_steps,timestep=timestep)         # creating fuel cell object
        available_hydrogen = 10e4                   # [kg] hydrogen available in the storage system

        hyd_used = np.zeros(sim_steps)      # [kg] hydrogen used by fuel cell
        P_el     = np.zeros(sim_steps)      # [kW] electricity produced
        P_th     = np.zeros(sim_steps)      # [kW] produced heat
        eta      = np.zeros(sim_steps)      # [-]  efficiency  
        water    = np.zeros(sim_steps)      # [Sm3/s]  produced water  
        
        flow = -np.random.uniform(0,fc.Npower,sim_steps)   # [kW] randomic power output as example
        
        for step in range(len(flow)):
            hyd_used[step],P_el[step],P_th[step],eta[step],water[step] = fc.use(step,flow[step],available_hydrogen)
        
#%%
        'Polarization Curve History'
        fig, ax = plt.subplots(dpi=1000) 
        
        for index,item in enumerate(fc.stack['Pol_curve_history']):
            ax.plot(fc.CellCurrDensity,item,label=f'{index}')
        
        ax.legend(title='Year',fontsize='small')
        ax.set_xlabel('Current density [A/cm$^{2}$]')
        ax.set_ylabel('Stack Voltage [V]')
        # ax.set_ylim(135.10,135.3)
        ax.grid(alpha=0.5,zorder=-1)    
        plt.show()
        
        'Efficiency History'
        fig, ax = plt.subplots(dpi=1000) 
        
        for index,item in enumerate(fc.stack['Module_efficiency[-]']):
            ax.plot(fc.CellCurrDensity,item,label=f'{index}')
        
        ax.legend(title='Year',fontsize='small')
        ax.set_xlabel('Current density [A/cm$^{2}$]')
        ax.set_ylabel('Module efficiency [-]')
        ax.grid(alpha=0.5,zorder=-1)    
        plt.show()
        
        'Conversion Factor Update'
        fig, ax = plt.subplots(dpi=1000)     
        ax.scatter(np.arange(sim_steps),fc.stack['Conversion_ratio_op[kWh/kg]'], s=0.1, label='operational')
        ax.scatter(np.arange(sim_steps),fc.stack['Conversion_ratio_rated[kWh/kg]'], s=0.1, label='rated')
        # ax.plot(np.arange(sim_steps),el.stack['Conversion_factor_rated[kg/MWh]'], linewidth=0.1, label='rated')
        # ax.plot(np.arange(sim_steps),el.stack['Conversion_factor_op[kg/MWh]'], linewidth=0.05, label='operational')
        ax.legend()
        ax.set_ylabel('Γ [kWh/kg]')
        ax.set_xlabel('Time [step]')
        ax.grid(alpha=0.5,zorder=-10)    
        plt.show()


        
