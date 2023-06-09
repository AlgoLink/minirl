from mlopskit import make

model_db = make("cache/uni_pricing_rl-v3", db_name="oneac.db")

# Agent parameters
agent_parameters = {
    "obs_dim": (34,),
    "action_n": 3,
    "actions": [],
    "eps": 0.2,
    "gamma": 0.99,
    "model_db": model_db,
    "score_db": None,
    "seed": 0,
    "alpha_w": 0.1,
    "alpha_theta": 0.1,
}
