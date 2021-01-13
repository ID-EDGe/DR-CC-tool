# Distributionally Robust Chance Constrained Optimization Tool for Distribution Grids
This flexible tool allows for operational planning including distributionally robust chance constraints for radial distribution grids. It was developed for a master thesis of Dominic Scotoni at DTU in collaboration with ETHZ. A documentation on how to use the tool is found in the *doc* folder. Sources are contained in the *src* folder. Results are exported automatically to the *rlt* folder and plots are stored in the *plt* folder. Below, see a more detailed list of the structure of the source folder. The requirements.txt file lists the packages required to run the program. Before running the program, parameters need to be set in the *param.py* file. Afterwards, *main.py* need so be executed to run the program.




### Sources: /src/ ...

    /src/caseN: case specific data, N indicating number of buses
    /src/fcst: PV forecast, automatically generated
    /src/load_profiles: normalized load profiles 
