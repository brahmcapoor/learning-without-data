from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tqdm import tqdm

from models.base_model import BaseModel


class StudentModel(BaseModel):

    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs, scope="student")
