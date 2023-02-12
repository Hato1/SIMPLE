from gym.envs.registration import register

register(
    id='Tableturf-v0',
    entry_point='tableturf.envs:TableturfEnv',
)

