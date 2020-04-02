# Layered-Integration-Approach-for-Multi-view-Analysis-of-Real-world-Datasets
This repository contains the code of the validation part of the paper "Layered-Integration-Approach-for-Multi-view-Analysis-of-Real-world-Datasets".

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
