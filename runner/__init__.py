from runner.sync_runner.sync_runner import sync_run
from runner.async_runner.train_runner import train_run
from runner.async_runner.sample_runner import sample_run

runner_REGISTRY = {}
runner_REGISTRY["train"] = train_run
runner_REGISTRY["sample"] = sample_run
runner_REGISTRY["sync"] = sync_run
