import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def _uncenter_boxes(boxes):
    '''convert from center coords to corner coords'''
    boxes[:, 0] -= boxes[:, 2]/2.
    boxes[:, 1] -= boxes[:, 3]/2.

def _get_class_labels(predicted_classes, class_list):
    labels = (predicted_classes).astype(int)
    labels = [class_list[l] for l in labels]
    return labels

def convert_yolo_detections_to_fiftyone(
    yolo_detections,
    class_list
    ):

    detections = []
    if yolo_detections.size == 0:
        return fo.Detections(detections=detections)

    boxes = yolo_detections[:, 1:-1]
    _uncenter_boxes(boxes)

    confs = yolo_detections[:, -1]
    labels = _get_class_labels(yolo_detections[:, 0], class_list)

    for label, conf, box in zip(labels, confs, boxes):
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=box.tolist(),
                confidence=conf
            )
        )

    return fo.Detections(detections=detections)

def get_prediction_filepath(filepath, run_number = 1):
    run_num_string = ""
    if run_number != 1:
        run_num_string = str(run_number)
    filename = filepath.split("/")[-1].split(".")[0]
    return f"runs/detect/predict{run_num_string}/labels/{filename}.txt"

def add_yolo_detections(
    samples,
    prediction_field,
    prediction_filepath,
    class_list
    ):

    prediction_filepaths = samples.values(prediction_filepath)
    yolo_detections = [read_yolo_detections_file(pf) for pf in prediction_filepaths]
    detections =  [convert_yolo_detections_to_fiftyone(yd, class_list) for yd in yolo_detections]
    samples.set_values(prediction_field, detections)


if __name__ == "__main__":
    # A name for the dataset
    name = "voxel51-dataset"

    # # The directory containing the dataset to import
    dataset_dir = "C:/Users/susan/Documents/_UM/23Fall/EECS504/cv_freshness/voxel51_data"

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
    )

    dataset.export(
        export_dir="my_yolo_dataset",
        dataset_type=fo.types.YOLOv5Dataset
    )

    # View the dataset in the App
    session = fo.launch_app(dataset, desktop=True)
    session.dataset = dataset
    session.wait()

    # filepaths = dataset.values("filepath")
    # prediction_filepaths = [get_prediction_filepath(fp) for fp in filepaths]
    # dataset.set_values(
    #     "yolov8n_det_filepath",
    #     prediction_filepaths
    # )

    # add_yolo_detections(
    #     dataset,
    #     "yolov8n",
    #     "yolov8n_det_filepath",
    #     coco_classes
    # )
