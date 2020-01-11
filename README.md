# ImageNet-Adversarial-attack
 I finished 14th in  the second adversarial attack competition in TianChi
 
**Environment**
* python=3.6
* tensorflow=1.12
* scipy=1.2

**conclusion**  
* baseline:[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)  
* trick1：在源代码的基础上再添加随机噪声、随机翻转、随机亮度调整，非常有用，可以提高迁移性能  
* trick2： 生成对抗样本后进行高斯滤波，非常有用  
* trick3：[SI-NI-FGSM](https://arxiv.org/abs/1908.06281)，每次迭代都循环几次将输入除以2^i，有用，但是速度慢  
* trick4：models ensemble 

**usage**
* you should download you dataset and put them in data/images
* you should download the models [here](https://github.com/tensorflow/models/tree/master/research/slim) and put them in models/
