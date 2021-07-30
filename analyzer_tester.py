from torchvision import models
from dnn_analyzer import ModelAnalyse

model = models.resnet18()
ModelAnalyse(model, (3, 244, 244), 1)