

# carla_yaw


End-to-end Autonomous Driving System using yaw information.

We implement a vision-based autonomous driving mode using yaw and localization based End-to-end autonomous driving model with pytorch.

Implemented Version
  - Python 3.6
  - Pytorch 1.3
  - Torchvision 0.2
  - CARLA 0.8.2
  

# Kinematic Vehicle Equation

We utilize the bicycle model form kinematic vehicle dynamics.
The yaw angle is calculated by Kinematic Vehicle Equation.

<img src="https://github.com/kimna4/carla_yaw/blob/main/Kinematic-bicycle-model-of-the-vehicle.png" width="90%"></img>

# How to Run
  - Train: python main_auxi_v0.py --lr=1e-4 --train-dir=your dir --eval-dir=your dir
  - Eval: python run_auxi_ver0.py --model-path=trained_model.pth


# CARLA benchmark
The CARLA simulator has a large variety of driving environments, such as traff ic lights and dynamic obstacles, including dynamic vehicles and pedestrians.

You can download the CARLA simulator from here ([benchmark]).

<img src="https://github.com/kimna4/carla_yaw/blob/main/carla.png" width="90%"></img>

[benchmark]: <https://carla.org/2018/04/23/release-0.8.2/>

licence : LGPL 2.1
