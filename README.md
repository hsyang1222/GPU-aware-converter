# GPU-aware-converter
System converting pytorch-based model network to minimize gpu.  
Currently optimized for changing unet.


# requirements
 
 - memcnn(https://memcnn.readthedocs.io/en/latest/installation.html). 
 - fastmri(https://github.com/facebookresearch/fastMRI)
   - install from git repository
   - visit https://fastmri.med.nyu.edu/ and submit Name, Email, Istitution
   - download dataset knee + singlecoil + validation
 
 - See requirements.txt for accurate versions and test environments.
 


# how to run example

 - If you want to run as a jupyer notebook file, run example.ipynb. That is the way we most recommend it.

 - If you want to run as pure python, run "python example.py".
