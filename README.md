# BenchNPIN: Benchmarking Non-prehensile Interactive Navigation
BenchNPIN is a comprehensive suite of benchmarking tools for mobile robot non-prehensile interactive navigation. The goal of BenchNPIN is to provide researchers a standarized platform for training and evaluating algorithms in non-prehensile interactive navigation. BenchNPIN provides simulated environments for a range of non-prehensile interactive navigation tasks, as well as implementations of established baselines. Further, BenchNPIN provides demonstration datasets for future imitation learning and behavior cloning research. 


## Build from Source

1. Clone the project

2. Install dependencies.
```bash
cd BenchNPIN
pip install -r requirements.txt
```

2. Install Gym environment
```bash
pip install -e .
```


### Running a simple environment
After following the installation steps, run
```bash
python tests/env_test.py
```
This script runs a straight-line policy and renders the visualization of the simulation.

### Running a demo for autonomous ship navigation using a planning-based policy
```bash
python tests/ship_ice_nav.py
```
This script runs the lattice planner for ship ice navigation

### Running a simple box delivery teleoperation pipeline
```bash
python tests/box_delivery_data_collection.py
```
This script runs a simple demonstration data collection pipeline on the box delivery environment.


### Train and evaluate baselines for autonomous ship navigation
```bash
python tests/ship_ice_baselines.py
```
