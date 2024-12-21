from super_gradients.training.transforms.keypoints import (
    KeypointsRandomHorizontalFlip,
    KeypointsHSV,
    KeypointsBrightnessContrast,
    KeypointsMosaic,
    KeypointsRandomAffineTransform,
    KeypointsLongestMaxSize,
    KeypointsPadIfNeeded,
    KeypointsImageStandardize,
    KeypointsImageNormalize,
    KeypointsRemoveSmallObjects
)

# Indexes of keypoints on the flipped image. When doing left-right flip, left hand becomes right hand.
#So this array contains order of keypoints on the flipped image. This is dataset specific and depends on
#how keypoints are defined in dataset.
#keypoints_random_horizontal_flip = KeypointsRandomHorizontalFlip(flip_index=config['flip_indexes'], prob=0.5)

keypoints_hsv = KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20)

keypoints_brightness_contrast = KeypointsBrightnessContrast(prob=0.5,
                                                            brightness_range=[0.8, 1.2],
                                                            contrast_range=[0.8, 1.2]
                                                            )

keypoints_mosaic = KeypointsMosaic(prob=0.8)

keypoints_random_affine_transform = KeypointsRandomAffineTransform(max_rotation=0,
                                                                   min_scale=0.5,
                                                                   max_scale=1.5,
                                                                   max_translate=0.1,
                                                                   image_pad_value=127,
                                                                   mask_pad_value=1,
                                                                   prob=0.75,
                                                                   interpolation_mode=[0, 1, 2, 3, 4]
                                                                   )

keypoints_longest_max_size = KeypointsLongestMaxSize(max_height=640, max_width=640)

keypoints_pad_if_needed = KeypointsPadIfNeeded(min_height=640,
                                               min_width=640,
                                               image_pad_value=[127, 127, 127],
                                               mask_pad_value=1,
                                               padding_mode='bottom_right'
                                               )

keypoints_image_standardize = KeypointsImageStandardize(max_value=255)

# keypoints_image_normalize = KeypointsImageNormalize(mean=[0.485, 0.456, 0.406],
#                                                     std=[0.229, 0.224, 0.225]
#                                                     )

keypoints_remove_small_objects = KeypointsRemoveSmallObjects(min_instance_area=1,
                                                             min_visible_keypoints=1
                                                             )


def define_set_transformations():
        """
        Defines Keypoints transformations for training/validation sets

        Returns:
        train_transforms (list): List of training set transformations

        val_transforms (list): List of validation set transformations
        """
        
        train_transforms = [
            keypoints_hsv,
            keypoints_brightness_contrast,
            keypoints_mosaic,
            keypoints_random_affine_transform,
            keypoints_longest_max_size,
            keypoints_pad_if_needed,
            keypoints_image_standardize,
            keypoints_remove_small_objects
        ]

        val_transforms = [
            keypoints_longest_max_size,
            keypoints_pad_if_needed,
            keypoints_image_standardize,
        ]

        return train_transforms, val_transforms