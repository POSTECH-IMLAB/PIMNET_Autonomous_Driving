carla_cil_pytorch_eval
===============

A pytorch implementation to evaluate the conditional imitation learning policy in "End-to-end Driving via Conditional Imitation Learning" and "CARLA: An Open Urban Driving Simulator".

Requirements
-------
pytorch > 0.4.0    
tensorboardX


Running
------
Start carla simulater and leave your trained policy weight in ***model/policy.pth***
run:
```
$ python run_CIL.py --log-name local_test --weathers 6 --model-path "model/policy.pth"
```

Policy Training
------
Please reference [carla_cil_pytorh](https://github.com/onlytailei/carla_cil_pytorch).    
For the benchmark results, please check our RA-L paper [VR-Goggles for Robots: Real-to-sim Domain Adaptation for Visual Control](https://ram-lab.com/file/tailei/vr_goggles/index.html).
