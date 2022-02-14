## Udacity Deep Reinforcement Learning Nanodegree
### Project 1: Navigation

##### &nbsp;

### Project Details
The goal of this project is to train a RL agent to navigate inside an square world collecting yellow bananas while avoiding blue ones.

The environment consists of a 37 dimensions state space which provides the agent's velocity along with an representation of the objects right in front of the agent. Moreover, the environment rewards the agent with +1 every time it collects a yellow banana, and -1 every time it collects a blue banana.

Every episode has a length of 300 steps, making it an episodic task.

The action space consists of four discrete actions presented here below:

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

Finally, the environment in considered solved when the agent manages to get an mean reward above 13 points in 100 episodes.

### Getting Started
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this folder, unzip (or decompress) the file and then write the correct path in the argument for creating the environment under the notebook `Navigation_solution.ipynb`:

```python
env = env = UnityEnvironment(file_name="Banana.app")
```

### Instructions
Follow the instructions in `Navigation.ipynb` to get started with training your own agent!
