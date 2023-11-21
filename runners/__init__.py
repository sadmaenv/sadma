from runners.train_runner import train_run
from runners.sample_runner import sample_run
from runners.sync_runner import sync_run

runner_REGISTRY = {}
runner_REGISTRY["train"] = train_run
runner_REGISTRY["sample"] = sample_run
runner_REGISTRY["sync"] = sync_run
