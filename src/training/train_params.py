from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
from super_gradients.training.utils.early_stopping import EarlyStop
    
KEYPOINT_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle"
]

NUM_JOINTS = len(KEYPOINT_NAMES) # Number of the keypoints 
OKS_SIGMAS = [0.07] * NUM_JOINTS # 

# The colors for individual keypoints
KEYPOINT_COLORS = [
    [0, 255, 0],    # nose
    [0, 0, 255],    # left eye
    [255, 0, 0],    # right eye
    [0, 255, 255],  # left ear
    [255, 0, 255],  # right ear
    [255, 255, 0],  # left shoulder
    [128, 0, 128],  # right shoulder
    [128, 128, 0],  # left elbow
    [0, 128, 128],  # right elbow
    [0, 128, 0],    # left wrist
    [128, 0, 0],    # right wrist
    [255, 128, 0],  # left hip
    [128, 255, 0],  # right hip
    [0, 255, 128],  # left knee
    [128, 0, 255],  # right knee
    [255, 0, 128],  # left ankle
    [255, 128, 128] # right ankle
]

# The links (or connections) between keypoints
EDGE_LINKS = [
    [0, 1], [0, 2], [1, 3], [2, 4],        # face
    [5, 6],                                # shoulders
    [5, 7], [7, 9], [6, 8], [8, 10],       # arms
    [5, 11], [6, 12],                      # torso
    [11, 12],                              # hips
    [11, 13], [13, 15], [12, 14], [14, 16] # legs
]

# Colors for the edge links
EDGE_COLORS = [
    [255, 0, 0],   # nose to left eye
    [0, 0, 255],   # nose to right eye
    [0, 255, 0],   # left eye to left ear
    [255, 255, 0], # right eye to right ear
    [255, 0, 255], # shoulders
    [0, 255, 255], # left shoulder to left elbow
    [128, 0, 128], # left elbow to left wrist
    [128, 128, 0], # right shoulder to right elbow
    [0, 128, 128], # right elbow to right wrist
    [255, 128, 0], # left shoulder to left hip
    [128, 255, 0], # right shoulder to right hip
    [0, 255, 128], # hips
    [128, 0, 255], # left hip to left knee
    [255, 0, 128], # left knee to left ankle
    [255, 128, 128], # right hip to right knee
    [255, 255, 128]  # right knee to right ankle
]


def define_train_params(NUM_EPOCHS = 5):
        """
        Defines training parameters

        Returns:
        Union[dict, list, None]: The content of the file parsed to a dictionary or a list,
                                or None if an error occurs.
        """
        
        try:
            post_prediction_callback = YoloNASPosePostPredictionCallback(
                pose_confidence_threshold = 0.01,
                nms_iou_threshold = 0.7,
                pre_nms_max_predictions = 300,
                post_nms_max_predictions = 30,
            )

            metrics = PoseEstimationMetrics(
            num_joints = NUM_JOINTS,
            oks_sigmas = OKS_SIGMAS,
            max_objects_per_image = 30,
            post_prediction_callback = post_prediction_callback,
            )

            visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
            keypoint_colors = KEYPOINT_COLORS,
            edge_colors = EDGE_COLORS,
            edge_links = EDGE_LINKS,
            loss_to_monitor = "YoloNASPoseLoss/loss",
            max = True,
            freq = 1,
            max_images = 1,
            enable_on_train_loader = True,
            enable_on_valid_loader = True,
            post_prediction_callback = post_prediction_callback,
            )

            early_stop = EarlyStop(
            phase = Phase.VALIDATION_EPOCH_END,
            monitor = "AP",
            mode = "max",
            min_delta = 0.0001,
            patience = 100,
            verbose = True,
            )

            train_params = {
                "warmup_mode": "LinearBatchLRWarmup",
                "warmup_initial_lr": 1e-8,
                "lr_warmup_epochs": 1,
                "initial_lr": 5e-5,
                "lr_mode": "cosine",
                "cosine_final_lr_ratio": 5e-3,
                "max_epochs": NUM_EPOCHS,
                "zero_weight_decay_on_bias_and_bn": True,
                "batch_accumulate": 1,
                "average_best_models": True,
                "save_ckpt_epoch_list": [5, 10, 15, 20],
                "loss": "yolo_nas_pose_loss",
                "criterion_params": {
                    "oks_sigmas": OKS_SIGMAS,
                    "classification_loss_weight": 1.0,
                    "classification_loss_type": "focal",
                    "regression_iou_loss_type": "ciou",
                    "iou_loss_weight": 2.5,
                    "dfl_loss_weight": 0.01,
                    "pose_cls_loss_weight": 1.0,
                    "pose_reg_loss_weight": 34.0,
                    "pose_classification_loss_type": "focal",
                    "rescale_pose_loss_with_assigned_score": True,
                    "assigner_multiply_by_pose_oks": True,
                },
                "optimizer": "AdamW",
                "optimizer_params": {
                    "weight_decay": 0.000001
                },
                "ema": True,
                "ema_params": {
                    "decay": 0.997,
                    "decay_type": "threshold"
                },
                "mixed_precision": True,
                "sync_bn": False,
                "valid_metrics_list": [metrics],
                "phase_callbacks": [visualization_callback, early_stop],
                "pre_prediction_callback": None,
                "metric_to_watch": "AP",
                "greater_metric_to_watch_is_better": True,
                "_convert_": "all"
            }

            return train_params   

        except Exception as e:
            
            print(f"An exception occurred in train_params.define_train_params(): {e}")
