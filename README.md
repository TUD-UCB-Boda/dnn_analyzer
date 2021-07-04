# DNN analyzer

Deep neural networks often consume a lot of computating power and memory, making them quite challenging to be deployed on edge devices.
Our **lightweight neural network analyzer based on PyTorch** predicts the computational requirements of a given DNN.

---
## Extracted Features

* total number of multiply–accumulate operations (MAC)
* total number of parameters
* inference memory
* memory read & write during inference
* disk storage
* duration ratio

---
## How to use

* Download the the DNN analyzer and unzip the folder
* Place the Python file that calls the analyzer in the folder dnn_analyzer
* The calling file must import the model_analysis file as below
* Start the analysis process by creating a new instance of ModelAnalyse passing the model to analyze and the input shape:
  model_analysis.ModelAnalyse(model, ([CHANNELS], [HEIGHT], [WIDTH]))

---
## Example

MobileNet_v2 analyzed with the dnn analyzer:

```python
from dnn_analyzer import model_analysis
from torchvision import models

model = models.mobilenet_v2()
model_analysis.ModelAnalyse(model, (3, 224, 224))
```
output:
```bash
+---------------+--------------+-----------------+------------------+----------------+------------------+---------------+---------------+
| module name   |   parameters |   mem read (MB) |   mem write (MB) |   storage (MB) |   inference (MB) |   MACs (Mega) | duration[%]   |
|---------------+--------------+-----------------+------------------+----------------+------------------+---------------+---------------|
| Conv2d        |          864 |           0.578 |            1.531 |          0.003 |            1.531 |         2.71  | 1.37 %        |
| BatchNorm2d   |           64 |           1.531 |            1.531 |          0     |            1.531 |         0.803 | 0.64 %        |
| ReLU6         |            0 |           1.531 |            1.531 |          0     |            1.531 |         0.401 | 0.66 %        |
| Conv2d        |          288 |           1.532 |            1.531 |          0.001 |            1.531 |         3.613 | 0.79 %        |
| BatchNorm2d   |           64 |           1.531 |            1.531 |          0     |            1.531 |         0.803 | 0.59 %        |
| ReLU6         |            0 |           1.531 |            1.531 |          0     |            1.531 |         0.401 | 0.46 %        |
| Conv2d        |          512 |           1.533 |            0.766 |          0.002 |            0.766 |         6.423 | 0.81 %        |
(... shortened to make the doc more readable ...)
| ReLU6         |            0 |           0.179 |            0.179 |          0     |            0.179 |         0.047 | 0.81 %        |
| Conv2d        |       307200 |           1.351 |            0.06  |          1.172 |            0.06  |        15.053 | 0.81 %        |
| BatchNorm2d   |          640 |           0.062 |            0.06  |          0.002 |            0.06  |         0.031 | 0.76 %        |
| Conv2d        |       409600 |           1.622 |            0.239 |          1.562 |            0.239 |        20.07  | 0.81 %        |
| BatchNorm2d   |         2560 |           0.249 |            0.239 |          0.01  |            0.239 |         0.125 | 0.81 %        |
| ReLU6         |            0 |           0.239 |            0.239 |          0     |            0.239 |         0.063 | 0.43 %        |
| Dropout       |            0 |           0     |            0     |          0     |            0.005 |         0     | 0.53 %        |
| Linear        |      1281000 |           4.887 |            0.004 |          4.887 |            0.004 |         1.28  | 0.66 %        |
+---------------+--------------+-----------------+------------------+----------------+------------------+---------------+---------------+
Total Storage:  13.37 MB
Total Parameters:  3504872
Total inference memory:  74.25 MB
Total number of Macs:  308.87 MMAC
Total Memory Read + Write:  162.19 MB
```
---
## Supported Layers

* ReLU, ReLU6, PReLU, LeakyReLU
* Conv1d, Conv2d, Conv3d
* MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d
* AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
* BatchNorm1d, BatchNorm2d, BatchNorm3d
* Linear

---
## Formulas used for calculations (not finished yet)

| Layer        | Computation | #parameters  | memory read | memory write | inference memory | disk storage |
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| FC      |  Cin x Hin x Win | (I + 1) × J  | Cin x Hin x Win x bpe* | (*if PReLU:* #params x ) Cin x Hin x Win x bpe* | [1*] | [2*] |
| conv      | K × K × Cin × (Hout / stride_y) × (Wout / stride_x) × (Cout / groups)  |   K × K × Cin × Cout | #params + Cout x Hout x Wout x bpe* | Cout x Hout x Wout x bpe* | [1*] | [2*] |
| pool   |   Cin x Hin x Win | (I + 1) × J  | Cin x Hin x Win x bpe* | Cout x Hout x Wout x bpe* | [1*] | [2*] |
| bn   |   Cin x Hin x Win ( x 2 *if learnable affine params*) | (I + 1) × J  | 2 * Cin + Cout x Hout x Wout x bpe* | Cout x Hout x Wout x bpe* | [1*] | [2*] |
| linear   |   inp x out | (I + 1) × J  | #params + Cout x Hout x Wout x bpe* | Cout x Hout x Wout x bpe*  | [1*] | [2*] |

bpe*: bytes per element,  [1*]: *Cout x Hout x Wout x bytes_per_elem*,  [2*]: *#params x bytes_per_param*

---
## Requirements

* tabulate > 0.8.0
* python > 3.6
* pytorch > 0.4.0
* NumPy > 1.14.3

---
## Authors

* [Jakob Michel Stock](https://github.com/Jeykobz) Student research assistant at the laboratory for Parallel Programming at TU Darmstadt
* [Arya Mazaheri](https://github.com/aryamazaheri) Research associate at the laboratory for Parallel Programming at TU Darmstadt
* [Tim Beringer](https://github.com/tiberi) Research associate at the laboratory for Parallel Programming at TU Darmstadt

---
## Some benchmarks

Model               | Input Resolution | Params(M) | Storage(MB) | inference memory(MB) | Memory Read+Write | MACs(G)     
---                 |---               |---        |---          |---          					|---								|---
alexnet							| (3, 224, 224)		 | 61,1			 | 233				 | 4,19									| 241,86						| 0,649
densenet121					| (3, 224, 224)		 | 7,98			 | 30,4				 | 147,1								| 359,71						| 2,79
densenet201					| (3, 224, 224)		 | 20 			 | 76,35			 | 219,59								| 581,5 						| 4,28
resnet18						| (3, 224, 224)		 | 11,69		 | 44,59			 | 28,53								| 102,88 						| 1,59
resnet50						| (3, 224, 224)		 | 25,56		 | 97,49			 | 122,2								| 342,89 						| 3,54
mobilenet_v2				| (3, 224, 224)		 | 3,5			 | 13,37			 | 74,25								| 162,19 						| 0,31
mobilenet_v3_small	| (3, 224, 224)		 | 2,54			 | 9,7				 | 16,2									| 35,46							| 0,054
mobilenet_v3_large	| (3, 224, 224)		 | 5,48			 | 20,92			 | 50,4									| 106,34						| 0,22
vgg11								| (3, 224, 224)		 | 132,9		 | 506,83			 | 62,69								| 632,59						| 7,62
vgg11_bn						| (3, 224, 224)		 | 132,9		 | 506,85			 | 91,02								| 689,27						| 7,64
vgg16								| (3, 224, 224)		 | 138,4		 | 527,79			 | 109,39								| 746,95						| 15,49
vgg16_bn						| (3, 224, 224)		 | 138,4		 | 527,82			 | 161,07								| 850,35						| 15,52
vgg19								| (3, 224, 224)		 | 143,7		 | 548,05			 | 119,34								| 787,12						| 19,65
vgg19_bn						| (3, 224, 224)		 | 143,7		 | 548,09			 | 176									| 900,47						| 19,68





---
## References

We have studied many different existing neural network analyzers to understand and benefit from their approaches.

Thanks to [@Swall0w](https://github.com/Swall0w) and [@sovrasov](https://github.com/sovrasov) who already implemented and published neural network analyzers.

* [flops-counter-pytorch](https://github.com/sovrasov/flops-counter.pytorch) -> we benefited from the initial version of the calculation of the computational requirements 
* [torchstat](https://github.com/Swall0w/torchstat) -> we took advantage of the initial version of the memory usage calculation and the approach of modifying the calling functions of the layers to be able to analyze them during inference

Other work from which we benefited:
* [How fast is my model?](https://machinethink.net/blog/how-fast-is-my-model/) -> blog post about predicting computational requirements of neural networks
* [Neural-Network-Analyser](https://github.com/rohitramana/Neural-Network-Analyser) -> neural network analyzer by [@rohitramana](https://github.com/rohitramana)
