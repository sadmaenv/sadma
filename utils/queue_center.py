from multiprocessing import Queue


# todo 增加超时检测
class QueueCenter:
    def __init__(self, args):
        self.local_mode = args.local
        self.evaluate_request_queue = Queue(10)
        self.infer_request_queue = Queue(args.num_sample_worker)
        self.shared_idx_queue = Queue(args.shared_buffer_size)
        self.log_queue = Queue(1000)

        if self.local_mode:
            self.sample_request_queue = [Queue(1) for _ in range(args.num_sample_worker)]
        else:
            self.sample_request_queue = Queue(args.num_sample_worker)

        self.shared_idx_queue = Queue(args.shared_buffer_size)
        for i in range(args.shared_buffer_size):
            self.shared_idx_queue.put(i)
        self.episode_idx_queue = Queue(args.shared_buffer_size)

        self.queue_dict = {
            "infer_request": self.infer_request_queue,
            "evaluate_request": self.evaluate_request_queue,
            "sample_request": self.sample_request_queue,
            "shared_idx": self.shared_idx_queue,
            "episode_idx": self.episode_idx_queue,
            "log": self.log_queue,
        }

    def get_sample_request(self, worker_id):
        assert self.local_mode
        return self.queue_dict["sample_request"][worker_id].get()

    def put_sample_request(self, worker_id):
        assert self.local_mode
        self.queue_dict["sample_request"][worker_id].put("sample")

    def get(self, queue_name):
        return self.queue_dict[queue_name].get()

    def put(self, queue_name, data):
        self.queue_dict[queue_name].put(data)

    def get_many(self, queue_name, number):
        result = []
        for _ in range(number):
            result.append(self.get(queue_name))
        return result

    def put_many(self, queue_name, datas):
        for data in datas:
            self.put(queue_name, data)
