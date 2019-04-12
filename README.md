# crf
pytorch 条件随机场实现序列标注

## 模型下载

[bert-vocab-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt)  
[bert-model-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)  
[Traditional-Chinese ELMo](http://vectors.nlpl.eu/repository/11/179.zip)

## Pre-requirements

* **must** python >= 3.6 (if you use python3.5, you will encounter this issue https://github.com/HIT-SCIR/ELMoForManyLangs/issues/8)
* pytorch 1.0
* opencc
* elmoformanylangs
* pytorch_pretrained_bert

## 简体转繁体

```
# 安装
# pip install opencc-python-reimplemented

# t2s - 繁体转简体（Traditional Chinese to Simplified Chinese）
# s2t - 简体转繁体（Simplified Chinese to Traditional Chinese）
# mix2t - 混合转繁体（Mixed to Traditional Chinese）
# mix2s - 混合转简体（Mixed to Simplified Chinese）

import opencc
cc = opencc.OpenCC('s2t')
s = cc.convert('你好，吃饭了吗？')
print(s)

```

## Result
>| Network + CRF        |    acc    |  precision  |    recall   |  f1-score  |
>|----------------------|-----------|-------------|-------------|------------|
>| Bi-GRU               |  0.9638   |   0.9294    |   0.9264    |   0.9279   |
>| 2 layer Bi-GRU       |  0.9613   |   0.9404    |   0.9362    |   0.9383   |
>| 3 layer Bi-GRU       |  0.9644   |   0.9366    |   0.9433    |   0.94     |
>| bert                 |  0.9655   |   0.9531    |   0.9556    |   0.9543   |
>| bert Bi-GRU          |  0.9648   |   0.9572    |   0.9511    |   0.9541   |
>| elmo Bi-GRU          |  0.9649   |   0.9468    |   0.9394    |   0.9431   |