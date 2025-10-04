# Machine Library for Non-Ideal Joints
This project provides a machine learning-based library for modeling non-ideal joints in mechanical systems. 
It aims to replace traditional joint models using pre-trained ML models.

## Features
- Types – Support for different non-ideal joint types (Generic joint)
- Properties – Flexibility, friction, contact, wear, tolerances, backlash, compliance, lubrication, fretting, misalignment, vibration    
- Exudyn – Data acquision and preprocessing tools (https://exudyn.readthedocs.io/en/v1.9.83.dev1/)
- PyTorch – Training and inference models (https://pytorch.org/)


## Dependencies 
- ngsolve (https://ngsolve.org/)
- cudatoolkit (https://developer.nvidia.com/cuda-toolkit)
- scikit-learn (https://scikit-learn.org/stable/)

## Folders and files
- Main file – 'MainScript.py' 
- Exudyn – Models/..
- ML –  ML/..
- Results – Solutions/.., ../data/, ../results/, ../MLmodels/
- Latex – Figures/.., References/.., 'LayOut.cls','Manuscript.tex'  

## License
- Open access? 

## Contact

- Qasim Khadim – University of Oulu (qasim.khadim@lut.fi)
- Andreas Zwölfer – Technical University of Munich (andreas.zwoelfer@tum.de)

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/qkhadim22/Non-Ideal-Joints-ML-Library.git