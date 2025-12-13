# Unifying-Day-and-Night-Tracking-with-Continual-Learning

### Data Preparation
**Put the tracking datasets in ./data. It should look like this:**
   ```
   ${UniTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- bdd100k_night
            |-- images
            |-- annotations
            ...
       
        -- shift_night
            |-- 0b3d-e686
            |-- 0b4d-d96f
            ...
       
   ```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Training
Dowmload the pretrained [[Baidu Drive]](https://pan.baidu.com/s/1pMc3SzshxhLTGTF99GrvMg?pwd=6wtc). 
### Testing
Download the model weights from  [BaiduNetDisk](https://pan.baidu.com/s/1l7objsrHt-NCMNs-b5AksQ) code: p647
