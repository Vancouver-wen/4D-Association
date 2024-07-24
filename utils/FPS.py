import os
import sys
import time

from loguru import logger

class FPS(object):
    def __init__(self) -> None:
        self.start=time.time()
        self.count=0
    def __call__(self, ) -> None:
        end=time.time()
        if end-self.start<1:
            self.count+=1
        else:
            logger.info(f"FPS:{self.count}")
            self.start=end
            self.count=0