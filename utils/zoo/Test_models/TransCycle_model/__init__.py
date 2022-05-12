import sys
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\TransCycle_model\axial_attention_module')
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\TransCycle_model')
import argparse

SUPPORTED_TASKS = ["segmentation", "classification", "detection"]

# def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
#
#     # classification network
#     parser = arguments_classification(parser=parser)
#
#     # detection network
#     parser = arguments_detection(parser=parser)
#
#     # segmentation network
#     parser = arguments_segmentation(parser=parser)
#
#     return parser

def get_model(opts):
    dataset_category = getattr(opts, "dataset.category", "segmentation")
    model = None
    # if dataset_category == "classification":
    #     model = build_classification_model(opts=opts)
    if dataset_category == "segmentation":
        model = TransCycle_model.TransCycle_model_30.model(arg=opts)
    # elif dataset_category == "detection":
    #     model = build_detection_model(opts=opts)
    else:
        task_str = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(dataset_category)
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        print(task_str)
    return model

