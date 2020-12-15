import os

server = 0
gpu_index = 2
eval = 0
config_files = []

# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_cf.yaml")

# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_SPRef_SPRet.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_VAR_SPRef_SPRet.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRet.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml")

config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_visible.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_orcnn_R_50_FPN_1x.yaml")
# config_files.append("D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal_visible.yaml")

# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_visible.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_orcnn_R_50_FPN_1x.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRet.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_cf_fm_AErefine.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_VAR_SPRef_SPRet.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_SPRef_SPRet.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet.yaml")
# config_files.append("KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml")

# config_files.append("COCOA_cls-AmodalSegmentation/mask_orcnn_R_50_FPN_1x.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_visible.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRet.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet.yaml")
# config_files.append("COCOA_cls-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml")

# config_files.append("COCOA-AmodalSegmentation/mask_orcnn_R_50_FPN_1x.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_visible.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_cf.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_cf_fm.yaml")
# config_files.append("COCOA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_cf_fm_AErecon.yaml")

checkpoint_path = "mask_rcnn_visible_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_amodal_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_d2sa_res50_SGD_1x_vdetach"
# checkpoint_path = "mask_rcnn_parallel_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_cf21_d2sa_res50_SGD_1x_v0"

# checkpoint_path = "mask_rcnn_parallel_CtRef_VAR_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_SPRef_SPRet_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_VAR_SPRef_SPRet_d2sa_res50_SGD_1x_2"
# checkpoint_path = "mask_rcnn_parallel_CtRef_VAR_SPRet_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef_VAR_SPRef_SPRet_d2sa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef_VAR_SPRef_SPRet_FM_d2sa_res50_SGD_1x"
# checkpoint_path = "orcnn_amodal_d2sa_res50_SGD_1x"

# checkpoint_path = "mask_rcnn_visible_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_cf21_kins_res50_SGD_1x_v3"
# checkpoint_path = "orcnn_amodal_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_cf21_fm34_recon7_kins_res50_SGD_1x_v3"
# checkpoint_path = "mask_rcnn_parallel_cf21_fm3_recon7_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_cf21_cosfm34_Arecls10_kins_res50_SGD_1x_v3"
# checkpoint_path = "mask_rcnn_parallel_CtRef0.5:0.5_VAR_SPRet_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef0.2:0.2_VAR0.05_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef0.5_VAR0.1_SPRef_SPRet_kins_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef0.5_VAR0.1_SPRef_SPRet_FM_KINS_res50_SGD_1x"

# checkpoint_path = "orcnn_amodal_cocoa_cls_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_c12_amodal_cocoa_cls_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_cocoa_res50_SGD_1x"
# checkpoint_path = "mask_rcnn_parallel_CtRef_VAR_SPRetc3_cocoacls_res50_SGD_1x"

for config_file in config_files:
    if server == 0:
        if not eval and gpu_index != -1:
            os.system('CUDA_VISIBLE_DEVICES={0} python tools/train_net.py '
                      '--config-file configs/{1}'
                      .format(gpu_index, config_file))
        elif eval and gpu_index != -1:
            os.system('CUDA_VISIBLE_DEVICES={0} python tools/train_net.py '
                      '--config-file configs/{1} '
                      '--eval-only MODEL.WEIGHTS /p300/workspace/detectron2/{2}/model_final.pth'
                      .format(gpu_index, config_file, checkpoint_path))
        elif not eval and gpu_index == -1:
            os.system('CUDA_VISIBLE_DEVICES=7,8 python tools/train_net.py '
                      '--num-gpus 2 --config-file configs/{0}'
                      .format(config_file))
    elif server == 1:
        if not eval and gpu_index != -1:
            os.system('CUDA_VISIBLE_DEVICES={0} python tools/train_net.py '
                      '--config-file configs/{1} OUTPUT_DIR /public/sist/home/xiaoyt/detectron2/workspace'
                      .format(gpu_index, config_file))
        elif eval and gpu_index != -1:
            os.system('CUDA_VISIBLE_DEVICES={0} python tools/train_net.py '
                      '--config-file configs/{1} '
                      '--eval-only MODEL.WEIGHTS /public/sist/home/xiaoyt/detectron2/{2}/model_final.pth'
                      .format(gpu_index, config_file, checkpoint_path))

