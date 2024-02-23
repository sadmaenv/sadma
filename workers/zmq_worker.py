import zmq
import numpy as np
from utils.utils import get_bs
from functools import partial
from threading import Thread
import json


class InferClient:
    def __init__(self, data_port, scheme, leader_address="localhost"):
        super(InferClient, self).__init__()
        self.scheme = scheme
        self._init_data_shape(scheme)

        ctx = zmq.Context()
        self.data_socket = ctx.socket(zmq.REQ)
        self.data_socket.connect(f"tcp://{leader_address}:{data_port}")

        self.poller = zmq.Poller()
        self.poller.register(self.data_socket, zmq.POLLIN)

    def _init_data_shape(self, data_scheme):
        send_data = []
        recv_data = []
        for k, v in data_scheme.items():
            info_tuple = (k, v['dtype'], v['vshape'])
            if v["zmq_send"] == "actor":
                send_data.append(info_tuple)
            if v["zmq_send"] == "learner":
                recv_data.append(info_tuple)
        self.send_dt = np.dtype(send_data)
        self.recv_dt = np.dtype(recv_data)

    def recv_data(self):
        sockets_poll = dict(self.poller.poll(500000))
        if self.data_socket in sockets_poll and sockets_poll[self.data_socket] == zmq.POLLIN:
            actions, info = self.data_socket.recv_multipart()
            info = json.loads(info.decode())
            actions = np.frombuffer(actions, dtype=self.recv_dt)[0][0].squeeze(-1)
            return actions, info["total_step"]
        else:
            return "stop", 0

    def send_data(self, datas, envs_step, alive_env, terminated_env_info, runner_id=0):
        info = {
            "runner_id": runner_id,
            "envs_step": envs_step.tolist(),
            "terminated_env_info": terminated_env_info,
            "alive_env": alive_env,
        }
        data_list = ()
        for name in self.send_dt.names:
            t = self.scheme[name]["dtype"]
            s = self.scheme[name]["vshape"]
            data_list += (datas[name].astype(t).reshape(s),)
        send_data = np.array([data_list], dtype=self.send_dt).tobytes()
        self.data_socket.send_multipart([send_data, json.dumps(info).encode()])

    def recv_send(self, datas, envs_step, runner_id=0):
        result_data = self.recv_data()
        self.send_data(datas, envs_step, runner_id)
        return result_data

    def send_recv(self, datas, envs_step, alive_env, terminated_env_info, runner_id=0):
        self.send_data(datas, envs_step, alive_env, terminated_env_info, runner_id)
        result_data = self.recv_data()
        return result_data


class InferServer:
    def __init__(self, args, episode_buffer, queue_center, data_scheme):
        self.args = args
        self.infer_batch_size = args.infer_batch_size
        self.get_bs = partial(get_bs, env_batch_size=args.env_batch_size, num_env_runner=args.num_env_runner)

        self.episode_buffer = episode_buffer
        self.queue_center = queue_center
        self.data_scheme = data_scheme
        self._init_data_shape(data_scheme)

        ctx = zmq.Context()
        self.data_sockets = []
        self.poller = zmq.Poller()
        for i in range(args.num_sample_worker):
            s = ctx.socket(zmq.REP)
            s.bind(f"tcp://{args.address}:{args.base_port + i}")
            self.data_sockets.append(s)
        for socket in self.data_sockets:
            self.poller.register(socket, zmq.POLLIN)

        self.stop_signal = False

    def _init_data_shape(self, data_scheme):
        send_data = []
        recv_data = []
        for k, v in data_scheme.items():
            info_tuple = (k, v['dtype'], v['vshape'])
            if v["zmq_send"] == "actor":
                recv_data.append(info_tuple)
            if v["zmq_send"] == "learner":
                send_data.append(info_tuple)
        self.send_dt = np.dtype(send_data)
        self.recv_dt = np.dtype(recv_data)

    def recv_data(self):
        self.queue_center.put("log", {"type": "info", "msg": "recv data thread start"})

        while not self.stop_signal:
            sockets_poll = dict(self.poller.poll(0))
            for sample_id, socket in enumerate(self.data_sockets):
                if socket in sockets_poll and sockets_poll[socket] == zmq.POLLIN:
                    recv_datas, info = socket.recv_multipart()
                    recv_datas = np.frombuffer(recv_datas, dtype=self.recv_dt)[0]
                    datas = {}
                    for idx, name in enumerate(self.recv_dt.names):
                        datas[name] = recv_datas[idx]
                    info = json.loads(info.decode())
                    runner_id = info["runner_id"]
                    ts = np.array(info["envs_step"])
                    terminated_env_info = info["terminated_env_info"]
                    alive_env = info["alive_env"]
                    if terminated_env_info:
                        self.queue_center.put("log", {"type": "data", "msg": terminated_env_info})
                    bs = self.get_bs(sample_id, runner_id)
                    if np.all(ts == 0):
                        self.episode_buffer.reset_data(bs)
                    self.episode_buffer.insert_data(datas, bs, ts)
                    infer_request = {
                        "infer_bs": bs,
                        "infer_ts": ts,
                        "sample_id": sample_id,
                        "alive_env": alive_env,
                    }
                    self.queue_center.put("infer_request", infer_request)

    def send_data(self):
        self.queue_center.put("log", {"type": "info", "msg": "send data thread start"})
        while not self.stop_signal:
            actor_id, bs, ts, total_step = self.queue_center.get("sample_request")
            info = {"total_step": total_step}
            data_list = ()
            for name in self.send_dt.names:
                data = self.episode_buffer[(name, bs, ts)]
                data_list += (data.numpy(),)
            send_data = np.array([data_list], dtype=self.send_dt).tobytes()
            self.data_sockets[actor_id].send_multipart([send_data, json.dumps(info).encode()])

    def start(self):
        self.ths = []
        recv_data_th = Thread(target=self.recv_data, daemon=True)
        self.ths.append(recv_data_th)
        send_data_th = Thread(target=self.send_data, daemon=True)
        self.ths.append(send_data_th)

        for t in self.ths:
            t.start()

    # todo safe kill thread
    def stop(self):
        self.stop_signal = True

    def wait(self):
        for t in self.ths:
            t.join()
