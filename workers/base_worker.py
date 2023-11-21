class BaseWorker:
    def __init__(self):
        self.job = None

    def start(self):
        self.job.start()

    def stop(self):
        if self.job:
            self.job.kill()

    def wait(self, timeout=None):
        if self.job:
            self.job.join(timeout=timeout)

    def terminate(self):
        if self.job:
            self.job.terminate()

    def exitcode(self):
        if self.job:
            return self.job.exitcode
        return None

    def get_pid(self):
        if self.job:
            return self.job.ident
        return None
