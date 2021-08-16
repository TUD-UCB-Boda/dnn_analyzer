from torchvision import models
from dnn_analyzer import ModelAnalyse

model = models.mobilenet_v2()
ModelAnalyse(model, (3, 244, 244), 1)