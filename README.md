# GPU-aware-converter
System converting pytorch-based model network to minimize gpu.  
Currently optimized for changing unet.


# run system demo
 - requirements
   - pytorch : conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   - memcnn : pip install memcnn
 - run plain model (without our system converter)
   - python demo-plain.py
 - run converted model with our system converter
   - python demo-our_system.py
 
# run full-type running
 - requirements
   - memcnn : https://memcnn.readthedocs.io/en/latest/installation.html 
   - fastmri : https://github.com/facebookresearch/fastMRI
   - See requirements.txt for environments and package version
 - download dataset
   - visit https://fastmri.med.nyu.edu/ and submit Name, Email, Istitution
   - download dataset knee + singlecoil + validation
 - run "example-fastmri.ipynb"

