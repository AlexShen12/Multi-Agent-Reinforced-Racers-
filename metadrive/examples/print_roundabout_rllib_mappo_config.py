from metadrive.envs.marl_envs.roundabout_rllib_delegator_env import RoundaboutRLLibDelegatorEnv 
from metadrive.utils.doc_utils import print_source

# THIS IS FOR TESTING THE CONFIG TO SEE IF IT MAKES SENSE
env = RoundaboutRLLibDelegatorEnv(
    {
        # "start_seed": 0,
        # "use_render": True,
        # "manual_control": True,
    }
)
env.print_config()
#print(env.action_spaces)
#print(env.observation_spaces)
# Observation spaces: Dict('agent0': Box(-0.0, 1.0, (91,), float32), 'agent1': Box(-0.0, 1.0, (91,), float32))
# Action spaces: Dict('agent0': Box(-1.0, 1.0, (2,), float32), 'agent1': Box(-1.0, 1.0, (2,), float32))
