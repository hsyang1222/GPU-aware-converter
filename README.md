# GPU-aware-converter
System converting pytorch-based model network to minimize gpu.  
Currently optimized for changing unet.


# run system demo
 - requirements
   - pytorch : conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   - memcnn : pip install memcnn
   - we test on python 3.8
 - see the effectiveness of a stitchable method
   - plain : python demo-stitchable.py --use_stitch 0
     - Upload images with sizes (64,4096,4096) and expect GPU assignment to be impossible to operate
   - stitchable(proposed) : python demo-stitchable.py --use_stitch 1
     - Upload images with sizes (64,4096,4096) and expect run slowly but successfully 
 - see the effectiveness of layer convert method
   - run plain model (without our system converter)
     - python demo-plain.py : require approximately 11GB
   - run converted model with our system converter
     - python demo-our_system.py : require approximately 4GB
 
# run full-type running
 - requirements
   - memcnn : https://memcnn.readthedocs.io/en/latest/installation.html 
   - fastmri : https://github.com/facebookresearch/fastMRI
   - See requirements.txt for environments and package version
 - download dataset
   - visit https://fastmri.med.nyu.edu/ and submit Name, Email, Istitution
   - download dataset knee + singlecoil + validation
 - run "example-fastmri.ipynb"

