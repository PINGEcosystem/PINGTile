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
    try:
        result = project.single_upload(
            image_path=image_path,
            annotation_path=annotation_filename,
        )
        # Roboflow returns a dict; check for an error key or a successful upload URL
        if result.get("error"):
            print(f"Upload failed for {image_path}: {result['error']}")
        else:
            print(f"Uploaded {image_path} -> {result.get('image') or result}")
    except Exception as e:
        print(f"Error uploading {image_path}: {e}")