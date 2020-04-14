# Layered-Integration-Approach-for-Multi-view-Analysis-of-Real-world-Datasets
This repository contains the code of the validation part of the paper "Layered-Integration-Approach-for-Multi-view-Analysis-of-Real-world-Datasets" which has been submitted to the ecml pkdd 2020 conference.

The data which has been used for this validation can be downloaded [here](https://opendata-renewables.engie.com/explore/index).

## Virtual Environment
You may be interested to have an anaconda virtual environment. This way, all necessary packages are installed on itself. In that case, you can use the environment_droplet.yml file.

1.Create the environment:

    conda env create -f environment_droplet.yml
    
    
2.Activate the environment: 

    conda activate ecmlpkdd_environment 

4.Open the notebook

     jupyter lab "Layered Integration Approach for Multi-view Analysis of Real-world Datasets.ipynb"

## Overview of the usable parameters in the Engie dataset:
Operational (endogenous) parameters:

- Pitch angle
- Generator bearing 1 temperature 
- Generator bearing 2 temperature
- Generator stator temperature
- Rotor bearing temperature
- Gearbox bearing 1 temperature
- Gearbox bearing 2 temperature
- Gearbox inlet temperature
- Gearbox oil sump temperature
- Nacelle temperature
- Nacelle angle*
- Nacelle angle corrected
- Rotor speed*
- Torque
- Generator converter speed*
- Generator speed*
- Converter torque*

Environmental (exogenous) parameters:
- First anemometer on the nacelle*
- Second anemometer on the nacelle*
- Average wind speed
- Absolute wind direction
- Absolute wind direction corrected
- Outdoor temperature
- Hub temperature (the hub height is the distance from the ground to the center-line of the turbine rotor)
- Grid frequency


Output (performance) parameters:
- Active power
- Reactive power
- Apparent power (Should be sqrt{P^2 + Q^2})
- Power factor (Should be P/S)

*These parameters are not used since they have an absolute correlation higher than 0.85.
