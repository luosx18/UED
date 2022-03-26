# UED

Code and data for "An Accurate Unsupervised Method for Joint Entity Alignment and Dangling Entity Detection".


## Usage
Training and local alignment:  
--mt_pair: use pseudo pairs  
--mt_anchor_weight: use globally guided loss  
--weight_decay: the weight decreasing mechanism of globally guided loss  
--data: path of data, e.g.: ./data/DBP15K or ./data/MedED  
--rate: #training_pairs / (#training_pairs + #testing_pairs)  
--cuda: gpu or cpu  
  
example 1:  
python UED.py --mt_pair --mt_anchor_weight --weight_decay --lang zh_en --data ./data/DBP15K --rate 0.3 --cuda  
example 2:  
python UED.py --mt_pair --mt_anchor_weight --weight_decay --lang es_en --data ./data/MedED --rate 0.3  
  
For the hyper-parameters, you can tune the hyper-parameters on the obtained pseudo entity pairs as the pseudo learning labels by generate more pseudo entity pairs with a lower threshold.   

  
Global alignment method and testing:  
python OTP.py --log_path train_logmt_pair_mt_anchor_weight_weight_decay_rate0.3 --lang zh_en --data ./data/DBP15K --rate 0.3  

  
## Datasets:
a. For DBP15K, we add mt_pair99, mt_sim_topK3 to the dataset we inherit from previous works (See ./data and our paper for details).   

b. For MedED, due to copyright of UMLS, if you need to obtain the real information of the entity and relationship, or construct KGs of other sizes, please download UMLS (version=2019ab, at https://www.nlm.nih.gov/research/umls/archive/archive_home.html), and refer to our data generation code in ./data  





## requirement：
apex  
pytorch >= 1.7.1  
torch_geometric >= 1.7.0  
torch_sparse >= 0.6.9  
mip >= 1.13.0  