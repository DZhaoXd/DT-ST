
####### dtst for G2C model adaptation
python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst.yaml OUTPUT_DIR results/dtst/ resume pretrain/G2C_model_iter020000.pth

#nohup python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst.yaml OUTPUT_DIR results/dtst/ resume pretrain/G2C_model_iter020000.pth > logs/dtst_g2c.file 2>&1 &


###### dtst for G2S model adaptation
python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst_synthia.yaml OUTPUT_DIR results/synthia_dtst/ resume ./pretrain/S2C_model_iter020000.pth

#nohup python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst_synthia.yaml OUTPUT_DIR results/synthia_dtst_scale_0.75_1.50_/ resume ./pretrain/S2C_model_iter020000.pth > logs/dtst_g2s.file 2>&1 &
