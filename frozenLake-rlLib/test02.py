import tensorflow as tf
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy

from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing

def postprocess_advantages(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    return compute_advantages(
        sample_batch, 0.0, policy.config["gamma"], use_gae=False)

def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) *
        train_batch[Postprocessing.ADVANTAGES])

MyTFPolicy = build_tf_policy(
    name="MyTFPolicy",
    loss_fn=policy_gradient_loss,
    postprocess_fn=postprocess_advantages)

import ray
from ray import tune
from ray.rllib.agents.trainer import Trainer

class MyTrainer(Trainer):
    def get_default_policy_class(self, config):
        return MyTFPolicy

ray.init()
tune.run(MyTrainer, config={"env": "CartPole-v0", "num_workers": 2})