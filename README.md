## Exploring Optimal Control of Epidemic Spread Using Reinforcement Learning

#### Authors: Abu Quwsar Ohi, M. F. Mridha, Muhammad Mostafa Monowar, and Md. Abdul Hamid

The overall script is implemented using Python-3.

The package requirenment are given in the "requirenments.txt" file.
Please execute "pip install -r requirenments.txt" to install the necessary packages.


The discription of the each file/folders are given below:

- **requirenments.txt:**     Contains necessary package/library names
- **agent_environment.py:** Contains the algorithm/implementation of the Virtual 
                          Environment, Agent, and necessary graph visualization 
                          implementations.
- **agent_training.py:**     Contains script to train an Agent using the Virtual 
                          Environment. Executing the script will further train a 
                          new agent. The new agent (along with the training variables) 
                          are repeteadly saved after 10 episodes in the "saved_params" 
                          folder. 
- **dashboard.py:**          Contains script that is used to generate graphs that 
                          visualize the environment and the actions performed by 
                          the agent. We used this script to generate the results of 
                          the agent. The script uses the model weights defined in the 
                          "base_model" directory.
- **base_model:**            The directory contains a TensorFlow generated Keras weights. 
                          The weights are generated through our training process defined 
                          in the paper.
- **train_logs:**            The directory contains a graphical report generated by the script 
                          "agent_training.py".
- **saved_params:**          The directory contains parameters and agent weights while executing 
                          the "agent_training.py" script. Erasing all files from this directory 
                          will cause the training to start from the beginning (Episode 1).
- **dashboard_logs:**        Contains graphical reports generated by the "dashboard.py" script.
