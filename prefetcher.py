from queue import Queue
import threading
from typing import Union
from torch.utils.data import DataLoader
import webdataset


class PrefetchDataLoader:
    def __init__(self, dataloader: Union[DataLoader, webdataset.WebLoader], prefetch_count: int = 1):
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count
        
    def __iter__(self):
        queue = Queue(maxsize=self.prefetch_count)
        def producer() -> None:
            for batch in self.dataloader:
                queue.put(batch)
            queue.put(None)
        thread = threading.Thread(
          target=producer,
          daemon=True # dies when main thread exits
        )
        thread.start()
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch
