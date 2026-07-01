"""v2 upgrade path (not implemented): full instance segmentation.

Swap the detection+segmentation path in MultiTaskModel for
`torchvision.models.detection.MaskRCNN` built on the *same*
`resnet_fpn_backbone` instance used in v1 (warm-startable from a v1
checkpoint):

    from torchvision.models.detection import MaskRCNN
    from torchvision.ops import MultiScaleRoIAlign

    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    model = MaskRCNN(backbone, num_classes=81, mask_roi_pool=mask_roi_pool)

This drops SemanticSegHead (Mask R-CNN gives per-instance masks, which is
strictly more information than the semantic mask). MultiLabelClsHead is
unaffected since it only reads backbone/FPN features, not detection output.

Why not v1: MaskRCNN bundles RPN + proposal sampling + RoIAlign + box/mask
heads inside its own GeneralizedRCNN.forward, which is not designed as an
extension point for splicing in a 4th (classification) task without
subclassing GeneralizedRCNN/RoIHeads internals. See the plan doc for the
full reasoning.
"""
