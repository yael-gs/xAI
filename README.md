# xAI
Exploration of Explainability models for AI Medical Image classification


# Use
To use the project, you need to install the requirements with the following command:
```bash 
pip install -r requirements.txt
```
Then look for the good version of torch torchvision and torchaudio according to your cuda version with the following command:
```bash
nvcc --version
```
You search for :
```bash 
Cuda compilation tools, release 12.6, V12.6.77 # "realease XX.X"
```
Then you can install the good version of torch with the following command and you replace the cuXXX with the good version of cuda, for example for cuda 12.6 you will replace cuXXX by cu126 and for cuda 11.8 you will replace cuXXX by cu118:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX #replace cuXXX by the good version of cuda
```
Then you can run GUI with : 
```bash 
streamlit run explain.py
```

Or 
run the scripts explainationMethods.py with the method you want to use on the data you want to use.