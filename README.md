# ReAFFINE
This repository contains Python code of the ReAFFINE method (Real-time Assessment of Flash Flood Impacts at pan-European Scale). This method can be used to estimate in real time the locations of severe flash flood impacts across Europe. The method is described and tested on two major flash flood events in the following publication: "Real-time assessment of flash flood impacts at pan-European scale: The ReAFFINE method", J. Ritter, M. Berenguer, S. Park, D. Sempere-Torres (2021), https://www.sciencedirect.com/science/article/pii/S0022169421010726 

The method is applied over Europe in 1 km resolution. Countries and EU Floods Directive flood maps included so far:
- Spain
- Germany
- Austria

The directory “static_layers” contains the preprocessed layers including the above listed countries. 

To launch ReAFFINE:
1. Install Python packages listed in requirements_reaffine.txt
2. Define input and output paths in config.properties (including the input flash flood hazard data in levels low-medium-high; see the above publication)
3. Run main program reaffine_v6.py
