import queue
import threading


class InferencePipeline:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._process_tasks)
        self.thread.daemon = True
        self.thread.start()

    def _process_tasks(self):
        while True:
            input_image, result_queue = self.task_queue.get()
            if input_image is None:
                break
            result = self.model_loader.predict(input_image)
            result_queue.put(result)
            self.task_queue.task_done()

    def add_task(self, input_image):
        result_queue = queue.Queue()
        self.task_queue.put((input_image, result_queue))
        return result_queue