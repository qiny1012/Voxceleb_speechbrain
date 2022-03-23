

#### 1. ECAPA-TDNN （C = 512）在Voxceleb 2 上训练，在Vox1__O，Vox1__E，Vox1__H上测试。

#### 2. ECAPA-TDNN （C = 512）+（多重注意力特征融合模块） 在Voxceleb 2 上训练，在Vox1__O，Vox1__E，Vox1__H上测试

#### **代码基于speechbrain音频处理包****

##### ****---训练---****

运行ECAPA-TDNN（512）原版的代码为：

`python train_speaker_embeddings.py 12_hparams_base.yaml --data_parallel_backend`

训练过程日志见文件`12_new.log`，显卡为3080\*2，训练时间约为40h

运行ECAPA-TDNN（512） + （多重注意力特征融合模块）的代码为：

`python train_speaker_embeddings.py 13_hparams_base.yaml --data_parallel_backend`

训练过程日志见文件`13_new.log`，显卡为3080*2，训练时间约为60h



常用的配置文件在 `/data/`路径下的压缩包中，可以通过调整文件路径使用。

训练过程中的配置文件说明：

```yaml
output_folder: ######/exp12 ## 日志输出位置
save_folder: !ref <output_folder>/save
train_log: !ref <output_folder>/train_log.txt

data_folder: ####### # 数据的位置，确保该路径下存在wav文件夹，该wav文件夹中包含vox2的数据
train_annotation: #####/data/train.csv ##训练集信息文件，按照speechbrain中生成
valid_annotation: #####/data/dev.csv ##验证集信息文件，按照speechbrain中生成

rir_folder: !ref <data_folder> # 噪声库位置，确保该路径下存在RIRS_NOISES文件夹
```

##### ****---测试---****

测试过程参考文件夹`12_test_code`和`13_test_code`



测试过程中配置文件说明：

```yaml
import sys
sys.path.append("######/notebook/model/") ## 引入模型代码
from ECAPA_TDNN import ECAPA_TDNN ## 原始ECAPA-TDNN模型结构

from ECAPA_TDNN_2 import ECAPA_TDNN ## ECAPA-TDNN + 多重注意力特征融合模块模型结构

params["output_folder"] = '######/exp12/" ##日志输出位置
params['pretrain_path'] = '######/exp12/save/CKPT+2022-03-19+04-49-19+00' ## 模型参数文件夹
params["train_data"] = "#####/data/train.csv" ##训练集信息文件，按照speechbrain中生成

params['enrol_data'] = "#####/data/enrol_O_.csv" ## vox1__O的注册语音信息
params['test_data']="#####/data/test_O.csv" ## vox1__O的测试语音信息veri_file_path = "#######/test_verif2.txt" ## 官方验证对

params["save_folder"] = os.path.join(params["output_folder"],"save") ## 结果输出位置params["voxceleb_source"] = "#######" ## 似乎没有使用
params["data_folder"] = "########" ## 测试集的地址，似乎没有什么用
```

##### ****---实验结果---****

ECAPA-TDNN（512）结果
Vox_O:    **1.02**    0.1038    0.1281
Vox_E:    1.27    0.1358    0.2498
Vox_H:    2.40    0.2257    0.4043



ECAPA-TDNN（512） + （多重注意力特征融合模块）结果

Vox_O:    **0.87**    0.1150    0.1809 
Vox_E:    1.22    0.1348    0.2519
Vox_H:    2.31    0.2223    0.3870




