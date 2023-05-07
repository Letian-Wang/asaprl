from gym.envs.registration import register, registry
from asaprl import SIMULATORS
# from .drive_env_wrapper import DriveEnvWrapper, BenchmkEnvWrapper
from .drive_env_wrapper import DriveEnvWrapper

envs = []
env_map = {}

if 'metadrive' in SIMULATORS:
    env_map.update({
        "Macro-v1": 'asaprl.envs.md_macro_env:MetaDriveMacroEnv',
    })

for k, v in env_map.items():
    if k not in registry.env_specs:
        envs.append(k)
        register(id=k, entry_point=v)

if len(envs) > 0:
    print("[ENV] Register environments: {}.".format(envs))
