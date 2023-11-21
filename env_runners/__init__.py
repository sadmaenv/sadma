runner_REGISTRY = {}
from .smac_runner import SMACRunner
from .replenishment_runner import ReplenishmentRunner

runner_REGISTRY["smac"] = SMACRunner
runner_REGISTRY["replenishment"] = ReplenishmentRunner
