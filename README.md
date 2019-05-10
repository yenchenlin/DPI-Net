# Accelerating DPI-Net

## Training
```
python train.py --env FluidShake --dataf /data/vision/torralba/tactile/physics_flex/data_FluidShake
```

## Pruning
```
CUDA_VISIBLE_DEVICES=3 python pruning.py --env RiceGrip --pruning_perc 99
```

## Evaluation
- FluidShake:
```
CUDA_VISIBLE_DEVICES=3 python eval.py --env FluidShake --epoch 4 --iter 500000 --dataf data/small/fluid_shake/ --model_file ./dump_FluidShake/files_FluidShake/net_epoch_4_iter_500000_pruning_95.pth
```

- BoxBath:
```
CUDA_VISIBLE_DEVICES=3 python eval.py --env BoxBath --epoch 4 --iter 370000 --dataf data/small/box_bath/ --model_file ./dump_BoxBath/files_BoxBath/net_epoch_4_iter_370000_pruning_95.pth
```

- RiceGrip:
```
CUDA_VISIBLE_DEVICES=3 python eval.py --env RiceGrip --epoch 18 --iter 130000 --dataf data/small/rice_grip/ --model_file ./dump_RiceGrip/files_RiceGrip/net_epoch_18_iter_130000_pruning_95.pth
```

- `pruning_perc`: `20` | `50` | `95` | `99`

## Misc 
Full data path:
```
/data/vision/torralba/tactile/physics_flex/data_FluidShake
```

---

### Install Dependencies
For Conda users, we provide an installation script:
```
bash ./scripts/conda_deps.sh
```
# Archive

## Evaluation

Go to the root folder of `DPI-Net`. You can direct run the following command to use the pretrained checkpoint.

    bash scripts/eval_FluidFall.sh
    bash scripts/eval_BoxBath.sh
    bash scripts/eval_FluidShake.sh
    bash scripts/eval_RiceGrip.sh

It will first show the grount truth followed by the model rollout. The resulting rollouts will be stored in `dump_[env]/eval_[env]/rollout_*`, where the ground truth is stored in `gt_*.tga` and the rollout from the model is `pred_*.tga`.


## Training

You can use the following command to train from scratch. **Note that if you are running the script for the first time**, it will start by generating training and validation data in parallel using `num_workers` threads. You will need to change `--gen_data` to `0` if the data has already been generated.

    bash scripts/train_FluidFall.sh
    bash scripts/train_BoxBath.sh
    bash scripts/train_FluidShake.sh
    bash scripts/train_RiceGrip.sh

## Citing DPI-Net

If you find this codebase useful in your research, please consider citing:

    @inproceedings{li2019learning,
        Title={Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids},
        Author={Li, Yunzhu and Wu, Jiajun and Tedrake, Russ and Tenenbaum, Joshua B and Torralba, Antonio},
        Booktitle = {ICLR},
        Year = {2019}
    }

