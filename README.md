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
 - run stitchable unit test
    ```bash 
      conda activate demo
      python demo-stitchable.py --use_stitch 1 # working
      python demo-stitchable.py --use_stitch 0 # OOM (not working)
    ```
    you can persistently monitor the GPU memory usage in separate terminal by
    ```
    watch -n0.2 nvidia-smi
    ```
 - run plain model (without our system converter)
    ```bash 
    conda activate demo
    python demo-plain.py
    ```
 - run converted model with our system converter
    ```bash
    conda activate demo
    python demo-our_system.py
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

