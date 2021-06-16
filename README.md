# GPU-aware-converter
System converting pytorch-based model network to minimize gpu.  
Currently optimized for changing unet.


# run system demo
 - requirements
   - memcnn : https://memcnn.readthedocs.io/en/latest/installation.html
   - pytorch
   - matplotlib
   - numpy
 - run "example-system_demo.ipynb"
 
# run full-type running
 - requirements
   - memcnn : https://memcnn.readthedocs.io/en/latest/installation.html 
   - fastmri : https://github.com/facebookresearch/fastMRI
   - See requirements.txt for environments and package version
 - download dataset
   - visit https://fastmri.med.nyu.edu/ and submit Name, Email, Istitution
   - download dataset knee + singlecoil + validation
 - run "example-fastmri.ipynb"

