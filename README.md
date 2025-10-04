# SLIDE networks to estimate structural dynamics
This project uses SLIDE networks for estimating structural dynamics in hydraulically actuated flexible multibody systems. It includes:
- An efficient data acquisition approach from random initial configurations
- SLIDE window computation without requiring equations of motion
- Validation across 1-DOF and 2-DOF configurations with different sensor setups and lifting loads
- Comparison against state-of-the-art models, including RNNs, LSTMs, and CNNs

## How to use it?
**Generate training data:** ()
- Setup – Choose an appropriate number of training and validation (test) simulations
- Models – 3D flexible (CMS-based) multibody systems, mode selection, hydraulics, materials (Steel, Aluminium, Titanium, and Composites), an algorithm to compute initial pressures from random initial configurations, 1-DOF (open-loop) and 2-DOF (closed-loop) multibody system configurations
- Exudyn – Data acquisition and preprocessing tools ([Exudyn Documentation](https://exudyn.readthedocs.io/en/v1.9.83.dev1/))

**Train SLIDE models:**
- Input – Valve control signals ($U_1, U_2$), actuator positions ($s_1, s_2$), actuator velocities ($\dot{s}_1, \dot{s}_2$), and hydraulic pressures ($p_1, p_2, p_3, p_4$)
- Output – Structural deflection ($\delta$), stress ($\sigma$), and strain ($\epsilon$)
- Preprocessing – Scaling and arranging data according to SLIDE network requirements
- PyTorch – Training and inference of SLIDE networks ([PyTorch](https://pytorch.org/))

## Dependencies 
- ngsolve (https://ngsolve.org/)
- cudatoolkit (https://developer.nvidia.com/cuda-toolkit)
- scikit-learn (https://scikit-learn.org/stable/)

## Folders and files
- Exudyn – models/..
- ML –  SLIDE/..
- solutions – .., ../data/, ../models/, ../Figures/


## References
- Use the references below for citing SLIDE original paper (1), its application in hydraulically flexible systems (2) and the hydraulic parameters (3). 
	1. Manzl, P., Humer, A., Khadim, Q. and Gerstmayr, J., 2025. SLIDE: A machine-learning based method for forced dynamic response estimation of multibody systems. Mechanics Based Design of Structures and Machines, pp.1-29.
	2. Khadim, Q., Manzl, P., Kurvinen, E., Mikkola, A., Orzechowski, G. and Gerstmayr, J., 2025. Real-time structural dynamics estimation in hydraulically actuated systems using 3D flexible multibody simulation and SLIDE networks. Mechanical Systems and Signal Processing, 240, p.113220.
 	3. Qasim Khadim, et al. Experimental investigation into the state estimation of a forestry crane using the unscented Kalman filter and a multiphysics model. Mechanism and Machine Theory, 189, 105405, 2023.


## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/qkhadim22/SLIDE-EstimatingStructuralDeflection.git