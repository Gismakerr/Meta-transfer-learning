# Meta-transfer-learning
实验思路及记录
# Data
选择Loess、Karst、Mountain、Desert、Flat、Hill、Glacier七种地貌类型的数据，每种地貌选择5个区域，分别切成64与500像素大小的小块。
# TfaSR测试
使用TfaSR预训练好的模型直接应用到七种地貌上，比较与双线性三次插值结果的坡度、高程精度指标的差异。
|  地貌   | TfaSR_RMSE_Slope  | TfaSR_RMSE_Elevation| Cubic_RMSE_Slope  | Cubic_RMSE_Elevation| Change_Elevation | Change_Slope | 
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| Loess  | 0 | 0 | 0 | 0 | 0 | 0 | 
| Karst  | 0 | 0 | 0 | 0 | 0 | 0 | 
| Hill   | 0 | 0 | 0 | 0 | 0 | 0 |
| Mountain | 0 | 0 | 0 | 0 | 0 | 0 | 
| Desert   | 0 | 0 | 0 | 0 | 0 | 0 | 
| Glacier  | 0 | 0 | 0 | 0 | 0 | 0 | 
| Plain    | 0 | 0 | 0 | 0 | 0 | 0 | 
# 元迁移测试（主实验）
Baseline: SRCNN，SRResNet，TfaSR
1、分别先在TfaSR数据集上进行预训练；
|  Loess  | Pre_SRCNN  | Pre_SRResNet | Pre_TfaSR  |  MF_SRCNN  | MF_SRResNet | MF_TfaSR |
|  ----   | ----  | ----  | ----  | ----  | ----  | ----  |
| RMSE_Ele_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Plain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Plain | 0 | 0 | 0 | 0 | 0 | 0 |
2、使用预训练好的模型直接应用到七种地貌上；
3、分别测试七种地貌，每次选六种地貌进行元迁移学习，在第七种地貌上进行测试。
#  讨论
1.与迁移学习的对比：每个地貌中，选取3个区域的数据做迁移性训练，剩下2个区域的数据做测试，保证迁移学习的更新次数与元迁移测试的次数相同；
|         | T_SRCNN  | T_SRResNet | T_TfaSR  |  MF_SRCNN  | MF_SRResNet | MF_TfaSR |
|  ----   | ----  | ----  | ----  | ----  | ----  | ----  |
| RMSE_Ele_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Plain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Plain | 0 | 0 | 0 | 0 | 0 | 0 |
2.消融实验：地形知识对模型性能的影响

|         | No_SRCNN  | No_SRResNet | No_TfaSR  |  MF_SRCNN  | MF_SRResNet | MF_TfaSR |
|  ----   | ----  | ----  | ----  | ----  | ----  | ----  |
| RMSE_Ele_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Ele_Plain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Loess | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Karst | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Hill | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Mountain | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Desert | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Glacier | 0 | 0 | 0 | 0 | 0 | 0 |
| RMSE_Slope_Plain | 0 | 0 | 0 | 0 | 0 | 0 |

3.地形复杂度与迁移性可能的联系。
