# GPU-aware-converter
System converting pytorch-based model network to minimize gpu.  
Currently optimized for changing unet.


# run system demo
 - requirements
   - make virtual environment as follow (python 3.8, pytorch 1.8.0, cuda 11.1, memcnn)
    ```bash
    conda create -n demo python=3.8 anaconda -y
    conda activate demo
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install memcnn
    ```
 - Note that we provide (1) the implementation with only-python (`master branch`) and (2) the implementation with `bind.so` to use cudnnConvolutionBackwardFilter (`cpp_binding branch`). If you want to run another, `git checkout master` or `git checkout cpp_binding`.
 - see the effectiveness of a stitchable method
   - plain : python demo-stitchable.py --use_stitch 0
     - Upload images with sizes (64,4096,4096) and expect GPU assignment to be fail
      ```bash 
      conda activate demo
      python demo-stitchable.py --use_stitch 0 # OOM (not working)
      ```
   - stitchable(proposed) : python demo-stitchable.py --use_stitch 1
     - Upload images with sizes (64,4096,4096) and expect run slowly but successfully 
      ```bash 
      conda activate demo
      python demo-stitchable.py --use_stitch 1 # working
      ```
 - see the effectiveness of layer convert method
   - run plain model (without our system converter)
      ```bash 
      conda activate demo
      python demo-plain.py # require approximately 11GB
      ```
   - run converted model with our system converter
      ```bash
      conda activate demo
      python demo-our_system.py # require approximately 4GB
      ```
 
# run full-type running
 - requirements
   - memcnn : https://memcnn.readthedocs.io/en/latest/installation.html 
   - fastmri : https://github.com/facebookresearch/fastMRI
   - See requirements.txt for environments and package version
 - download dataset
   - visit https://fastmri.med.nyu.edu/ and submit Name, Email, Istitution
   - download dataset knee + singlecoil + validation
 - run "example-fastmri.ipynb"

