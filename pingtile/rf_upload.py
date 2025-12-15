import glob
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="ADD_ROBOFLOW_API_KEY_HERE") #More info: https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key

# Directory path and file extension for images
dir_name = r"Z:\tmp\pingtile_test\12_12\json"
file_extension_type = ".png"

# Annotation file path and format (e.g., .coco.json)
annotation_filename = r"Z:\tmp\pingtile_test\12_12\json\_annotations.coco.json"

# Get the upload project from Roboflow workspace
project = rf.workspace().project("ADD_PROJECT_NAME_HERE")

# Upload images
image_glob = glob.glob(dir_name + '/*' + file_extension_type)
for image_path in image_glob:
    print(project.single_upload(
        image_path=image_path,
        annotation_path=annotation_filename,
        # -- optional parameters: --
        # annotation_labelmap=labelmap_path,
        # split='train',
        # num_retry_uploads=0,
        # batch_name='batch_name',
        # tag_names=['tag1', 'tag2'],
        # is_prediction=False,
    ))