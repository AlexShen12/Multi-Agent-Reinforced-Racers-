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