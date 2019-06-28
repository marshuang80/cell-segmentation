from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np

class Logger(object):

    def __init__(self, log_dir):
        """TensorboardX wrapper

        Parameters
        ----------
            log_dir: str 
                path to the log directory
        """

        # check if logger directory exist
        if not os.path.exists('logs'):
            os.mkdir('./logs')

        # include time to log file name
        self.log_dir = os.path.join('./logs', log_dir + '_{}.log'.format(datetime.now().strftime('%b%d_%H%M')))

        # create tensorboardX summary writer
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
   

    def update_value(self, value_label, value, step):
        """Update numeric values on TensorboardX

        Parameters
        ----------
            value_label: str
                label name for the logged value
            value: (int, float)
                value to be logged 
            step:
                the step/epoch/itr of the logged value
        """

        self.summary_writer.add_scalar(value_label, value, step)
   

    def update_image(self, label, img, step):
        """Update images on TensorboardX

        Parameters
        ----------
            label: str
                label name for the logged image
            img: array
                an matrix representation of image to be logged
            step:
                the step/epoch/itr of the logged image
        """

        self.summary_writer.add_image(label, img, step) 
