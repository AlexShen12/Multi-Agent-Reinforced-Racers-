# Multi-Agent Reinforced Racers: Training Multiple Agents to Race in MetaDrive with RLLib
## 🛠 Quick Start
Install MetaDrive via:

```bash
git clone https://github.com/AlexShen12/Multi-Agent-Reinforced-Racers- --branch JAMES-MAPPO --single-branch <folder>

cd metadrive
pip install -e .
```

You can verify the installation of MetaDrive via running the testing script:

```bash
# Go to a folder where no sub-folder calls metadrive
python -m metadrive.examples.drive_in_safe_metadrive_env.py
```
Press ESC to quit, T to take control of the car from the expert agent, or H to help.
*Note that please do not run the above command in a folder that has a sub-folder called `./metadrive`.*

## Multi-Agent Training with MetaDrive and RLLib
To train a Multi-Agent Reinforcement Learning algorithm in our Multi-Agent Roundabout environment with RLLib:
1. Locate the Jupyter Notebook metadrive.examples.roundabout_rllib_notebook.ipynb
2. Install RLLib's dependencies
    ```bash
    pip install "ray[rllib]" torch
    ```
3. In the first cell of the Jupyter Notebook, update the base filepath in the ```checkpoint_base_directory``` string. Later cells will create a folder at the given ```checkpoint_base_directory + str(checkpoint_number)``` filepath to store the RLLib Algorithm training information. A top level "Checkpoints" folder with a subfolder "MappoCheckpoint" will lead to the creation of Checkpoints/MappoCheckpoint0, Checkpoints/MappoCheckpoint1, etc. as the training saves the Algorithm state every 50 episodes.
4. Follow cells 1-5 to perform headless training and save Algorithm state every 50 episodes. If the 4th cell (for building the configuration for the RLLib PPO Algorithm specified by cell 3) fails, update the 3rd cell by commenting out .env_runners() and .learners() and uncommenting the commented .env_runners() and .learners(). This will use less compute resources.
5. As soon as you have trained for 50 iterations/episodes, you can test out the model! Head to the next section. You can leave the Jupyter Notebook running to continue creating more checkpoints while you test the latest checkpoint.

## Multi-Agent Testing with MetaDrive
To test a trained MARL algorithm in a rendered Multi-Agent Roundabout environment:
1. Locate the Python file metadrive.examples.drive_in_roundabout_rllib_mappo_env.py
2. Update ```checkpoint_base_directory``` and ```checkpoint_number``` to use the latest trained checkpoint. Checkpoint 0 is untrained.
3. Run with ```python -m metadrive.examples.drive_in_roundabout_rllib_mappo_env```.

## Acknowledgement
The simulator can not be built without the help from Panda3D community and the following open-sourced projects:
- panda3d-simplepbr: https://github.com/Moguri/panda3d-simplepbr
- panda3d-gltf: https://github.com/Moguri/panda3d-gltf
- RenderPipeline (RP): https://github.com/tobspr/RenderPipeline
- Water effect for RP: https://github.com/kergalym/RenderPipeline 
- procedural_panda3d_model_primitives: https://github.com/Epihaius/procedural_panda3d_model_primitives
- DiamondSquare for terrain generation: https://github.com/buckinha/DiamondSquare
- KITSUNETSUKI-Asset-Tools: https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools
