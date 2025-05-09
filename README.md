# Imitation Learning for Car Racing Environment
Final project of course EECE 571S - Machine Teaching at UBC

## Course
EECE 571S - Machine Teaching @ UBC

## Semester
2024 Winter Session Term 2

## Outline
This is the repo for the final project of the course EECE 571S. This project aims to use `PPO`, `behavioral cloning`, and `DAgger` to train agents through a 2D car racing simulation environment provided by OpenAI, making the agent successfully navigate the vehicle throughout the race track. Through the experiments and results, we would like to discuss how imitation learning can be effective for this learning task through our implementation. We also want to discuss the moving patterns for `continuous` and `discrete` action spaces. We successfully trained the expert policy as a demonstration source to further train other policies with the imitation learning algorithms. 

All best models surpass `85%` of the baseline performance, while only having the time cost lower than `10%` of the baseline. The best policy with the highest performance is trained through DAgger in discrete action space, reaching `94.7%` of the PPO policy performance. We found how efficient the BC and DAgger algorithm can be, compared to the time cost of the PPO algorithm. The BC policies in continuous and discrete settings can have less than `1%` of the baseline time cost, while both still preserving over `85%` of the PPO policy performance. 

## Result and Demo
![pixel-style](https://raw.githubusercontent.com/FalKon1256/UBC-EECE-571S
/main/imgs/results.png)

This figure shows the final performance, improvement, and trade-off comparison of our training. The final report and results are shown in the [final project report](https://github.com/FalKon1256/UBC-EECE-571S/blob/main/report/EECE_571S-final-project-report-kevinchu.pdf). You may find the demos from [this link](https://www.youtube.com/playlist?list=PLr1LfzWLnxUCpO9GbtjgPFEKumtLGQ36u). 


## Get Started

### Installation
1. Clone this repo. 
```bash
git clone https://github.com/FalKon1256/UBC-EECE-571S.git
```

2. Create the environment using conda (the code is based on Windows 11, there may be errors if you are running on another OS). 
```bash
conda create -n my-bc python=3.10 -y

pip install gymnasium
pip install pygame
pip install swig
pip install "gymnasium[box2d]"
pip install opencv-python
pip install stable-baselines3[extra]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # use the correct version of cuda for your system
pip install moviepy
pip install imitation
pip install imitation[test]
```

### Usage
1. Train `PPO` expert policy. 
```bash
  # Please modify the control parameters for your experiments. 
  # Then run the ppo_train.py file. 
  python ppo_train.py

  # The checkpoints are saved in the "models\ppo" folder, for example: 
  models\ppo\continuous\ppo_timesteps_1000000\checkpoints\ppo_continuous_1000000_steps.zip
  models\ppo\discrete\ppo_timesteps_1000000\checkpoints\ppo_discrete_1000000_steps.zip
```

2. Use PPO-trained policy to implement `BC` training. 
```bash
  # Please modify the control parameters for your experiments. 
  # Then run the bc_train.py file. 
  python bc_train.py

  # The policies are saved in the "models\bc" folder, for example: 
  models\bc\bc-continuous-traj_20.zip
  models\bc\bc-discrete-traj_20.zip
```

3. Use PPO-trained policy to implement `DAgger` training. 
```bash
  # Please modify the control parameters for your experiments. 
  # Then run the dagger_train.py file. 
  python dagger_train.py

  # The policies are saved in the "models\dagger" folder, for example: 
  models\dagger\dagger-continuous-totalsteps_10000.zip
  models\dagger\dagger-discrete-totalsteps_10000.zip
```

4. Load a trained policy and run the environment. 
```bash
  # Please modify the control parameters for your experiments. 
  # Then run the run_trained_policy.py file. 
  # If you want to load a specific policy, please modify the path. 
  python run_trained_policy.py

  # If you use the special evaluation mode, 
  # you will be able to get the action distribution data. 
  SEE_BEST_MODEL_ACTIONS = True  # Set this to True
```

5. Check experiment data. 
```bash
  # The log data will be in the "logs\" folder, 
  # please use Tensorboard to check or save the raw data. 
  logs\ppo_log
  logs\bc_log
  logs\dagger_log

  # For the action distribution data, please open the .txt file directly. 
  logs\action_list
```

## Acknowledgement

This work is based on many amazing libraries, thanks a lot to all the authors for sharing!

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [Imitation](https://imitation.readthedocs.io/en/latest/index.html#)
- [OpenAI’s Gymnasium - Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/)
