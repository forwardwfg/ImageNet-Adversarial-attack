# Improved-TI-attack
 I finished 14th in  the second adversarial attack competition in TianChi

baseline:Translation-Invariant-Attacks  
trick1：在源代码的基础上再添加随机噪声、随机翻转、随机亮度调整，非常有用，可以提高迁移性能  
trick2： 生成对抗样本后进行高斯滤波，非常有用  
trick3：SI-NI-FGSM，每次迭代都循环几次将输入除以2^i，有用，但是速度慢  
trick4：模型集合  
