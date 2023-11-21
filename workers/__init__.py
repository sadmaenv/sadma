from workers.sample_worker import LocalSampleWorker, RemoteSampleWorker
from workers.infer_worker import InferWorker
from workers.train_worker import TrainWorker
from workers.evaluate_worker import EvaluateWorker
from workers.zmq_worker import InferClient, InferServer

worker_REGISTRY = {}
worker_REGISTRY["local_sample_worker"] = LocalSampleWorker
worker_REGISTRY["remote_sample_worker"] = RemoteSampleWorker
worker_REGISTRY["infer_worker"] = InferWorker
worker_REGISTRY["train_worker"] = TrainWorker
worker_REGISTRY["evaluate_worker"] = EvaluateWorker
worker_REGISTRY["infer_client"] = InferClient
worker_REGISTRY["infer_server"] = InferServer
