from dataclasses import dataclass
from loguru import logger
import numpy as np

class Tracker:
    def __init__(self,*,optimizer_name:str,phase:str) -> None:
        self.optimizer_name = optimizer_name
        self.phase = phase
        self.records = []
        self.epoch = None

    def spawn_record(self):
        tr = TrackingRecord(self)
        tr.epoch = self.epoch
        return tr
    
    def save_records(self,target_dir):
        np.save(f"{self.optimizer_name}-{self.phase}.npy",self.records)
    

class TrackingRecord:
    def __init__(self,tracker:Tracker) -> None:
        self.tracker = tracker
        self._data = {"epoch":None,"batch":None,"loss":None}

    @property
    def epoch(self):
        return self._data["epoch"]

    @epoch.setter
    def epoch(self,e):
        logger.debug("set epoch")
        self._data["epoch"] = e
        logger.debug(sum(map(lambda value: value != None,self._data.values())))

        if sum(map(lambda value: value != None,self._data.values())) == 3:
            self.tracker.records.append(self._data)
            logger.debug("send data")


    @property
    def batch(self):
        return self._data["batch"]

    @batch.setter
    def batch(self,b):
        logger.debug("set batch")

        self._data["batch"] = b 
        logger.debug(sum(map(lambda value: value != None,self._data.values())))

        if sum(map(lambda value: value != None,self._data.values())) == 3:
            self.tracker.records.append(self._data)
            logger.debug("send data")

    @property
    def loss(self):
        return self.data["loss"]

    @loss.setter
    def loss(self,l):
        logger.debug("set loss")
        self._data["loss"] = l 

        logger.debug(sum(map(lambda value: value != None,self._data.values())))
        logger.debug(self._data.values())

        print(self._data)
        if sum(map(lambda value: value != None,self._data.values())) == 3:
            self.tracker.records.append(self._data)
            logger.debug("send data")



if __name__ == "__main__":
    t = Tracker(optimizer_name="adam",phase="train")
    tr = t.spawn_record()
    tr.loss = 1000
    tr.epoch = 1
    tr.batch = 100
    print(tr)