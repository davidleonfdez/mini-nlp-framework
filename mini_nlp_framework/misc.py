from abc import ABC, abstractmethod


class ProgressTracker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update_progress(self, i:int):
        "Method that receives a notification of the completion of the event with index `i`"
