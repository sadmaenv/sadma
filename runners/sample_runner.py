from workers import worker_REGISTRY


def sample_run(args):
    assert not args.local
    workers = []
    for i in range(args.num_sample_worker):
        sample_worker = worker_REGISTRY["remote_sample_worker"](args, worker_id=args.sampler_id + i)
        workers.append(sample_worker)
    for worker in workers:
        worker.start()
