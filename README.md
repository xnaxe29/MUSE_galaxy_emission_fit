# MUSE_galaxy_emission_fit
Fitting galaxy emission lines for MUSE 3D IFU data

File Description - 
1. 'basic_dictionary.dat' - Dictionary File for fitting. One can copy this file and change it accordingly and feed it as an argument to the python code 
from the terminal. Say the new dictionary file is test.dat, then the file 'fit.py' can be run with the new dictionary as - "python fit.py test.dat" 
entered to the terminal

2. 'fit.py' - Fitting the central spaxel as given in the dictionary
3. 'make_table.py' - Making table retrieving information from 'fit.py' saved result file
4. 'save_fig.py' - Saving Figure for the fit performed with 'fit.py'
5. 'muse_fit.py' - Fitting all MUSE spaxels as given in the dictionary
6. 'muse_save_new_table.py' - The initial fit gives a small undescribed data for the fit results. This code expands the fit results from 'muse_fit.py'
to make a detailed descriptive data table with proper information. (note to self: Add more details about this)
7. 'snr_map.py' - get the signal to noise (of H-alpha region, have to be described within code) of all emission based spaxels
8. 'binned_muse_fit.py' - Bin the spaxels of muse in binXbin square cubes, for a square bin-size of length, width - 'bin' val
9. 'custom_functions.py' - Custom functions that is called from all other python codes  
