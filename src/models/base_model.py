import copy
import torch
from .predictor import get_predictor


class BaseModel:
    """
    模型预测的基类
    """

    def __init__(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)
        self.predictor = get_predictor(**self.kwargs)
        self.device = torch.cuda.current_device()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.predict_type = kwargs.get("predict_type", "trt")

        if self.predictor is not None:
            self.input_shapes = self.predictor.input_spec()
            self.output_shapes = self.predictor.output_spec()

    def input_process(self, *data):
        """
        输入预处理
        :return:
        """
        pass

    def output_process(self, *data):
        """
        输出后处理
        :return:
        """
        pass

    def predict(self, *data):
        """
        预测
        :return:
        """
        pass

    def __del__(self):
        """
        删除实例
        :return:
        """
        if self.predictor is not None:
            del self.predictor
