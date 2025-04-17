
 # RL Based Car Navigation : DQN vs PPO Comparision
 
 A collection of Deep Reinforcement Learning agents for the OpenAI Gymnasium `CarRacing-v2` environment, including DQN variants (Vanilla DQN, Double DQN, Dueling DQN, PER DQN) and PPO (via Stable-Baselines3).
 
 ## Features
 
 - **DQN Family**:  
   - Vanilla DQN with ε-greedy, target network, replay buffer  
   - Double DQN  
   - Dueling DQN  
   - Prioritized Experience Replay DQN (PER DQN)
 
 - **PPO**:  
   - Proximal Policy Optimization using Stable-Baselines3
 
 - **Utilities**:  
   - Configurable preprocessing (frame resizing, grayscaling, stacking)  
   - Logging and performance metrics  
   - Training, evaluation, video recording, and plotting scripts  
   - Ready for Kaggle or local GPU execution
 
 ## Installation
 
 ```bash
 # Clone the repository
 git clone https://github.com/nc/rl-carracing-dqn-vs-ppo.git
 cd car_racing_rl
 
 # Create and activate a virtual environment (optional)
 python3 -m venv .venv
 source .venv/bin/activate
 
 # Install dependencies
 pip install -e .
 ```
 
 ## Configuration
 
 All hyperparameters are in `config.yaml`. You can adjust:
 
 - **Environment**: frame width, height, stack size, action space  
 - **Training**: number of episodes, steps per episode, replay frequency, seeds  
 - **DQN & PER**: learning rate, gamma, batch size, memory size, epsilon schedule  
 - **PPO**: learning rate, n_steps, batch size, n_epochs, clip range, etc.  
 - **Evaluation**: number of evaluation episodes, max steps, video recording length
 
 ## Usage
 
 ### Training
 
 ```bash
 # Train a single algorithm with a specific seed
 python -m scripts.train \
   --algo DQN \
   --seed 0 \
   --model_dir models \
   --log_dir logs
 ```
 
 Or using installed console script:
 
 ```bash
 car-racing-train --algo DQN --seed 0
 ```
 
 ### Evaluation
 
 ```bash
 python -m scripts.evaluate \
   --algo DQN \
   --model_path models/DQN_<timestamp>/DQN_final.weights.h5
 ```
 
 ### Record Video
 
 ```bash
 python -m scripts.record_video \
   --algo DQN \
   --model_path models/DQN_<timestamp>/DQN_final.weights.h5 \
   --video_dir videos
 ```
 
 ### Plot Metrics
 
 ```bash
 python -m scripts.plot_metrics \
   --eval_dir logs \
   --q_hist_dir logs \
   --out_dir figures
 ```
 
 ## Folder Structure
 
 ```
 .
 ├── algos/                # DQN & PPO agent implementations
 ├── envs/                 # Environment wrapper
 ├── scripts/              # Training, evaluation, video & plotting scripts
 ├── config.yaml           # Hyperparameter and environment configuration
 ├── requirements.txt      # Core dependencies
 ├── setup.py              # Package descriptor
 └── README.md             # Project overview and instructions
 ```
 
 ## License
 
 This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
