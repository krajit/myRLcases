
from ray.rllib.examples.env.mbmpo_env import CartPoleWrapper
from ray import tune

config = {
        "env":CartPoleWrapper,  
       "framework": "torch",
        #horizon: 200,
        "num_envs_per_worker": 20,
        "inner_adaptation_steps": 1,
        "maml_optimizer_steps": 8,
        "gamma": 0.99,
        "lambda": 1.0,
        "lr": 0.001,
        "clip_param": 0.5,
        "kl_target": 0.003,
        "kl_coeff": 0.0000000001,
        "num_workers": 8,
        "num_gpus": 0,
        "inner_lr": 0.001,
        "clip_actions": False,
        "num_maml_steps": 15,
        "model": {
            "fcnet_hiddens": [32, 32],
            "free_log_std": True,
        }        
}

stop = {
   "episode_reward_mean": 190,
   "training_iteration": 20
}

analysis =  tune.run(
    "MBMPO",
    config=config,
    stop=stop,
    checkpoint_at_end=True,  
    checkpoint_freq=5,  
    local_dir="checkPoints/",
    verbose=1,
)