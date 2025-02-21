# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_large_folder(
#     repo_id="moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.10_channel",
#     repo_type="model",
#     folder_path="DeepSeek-R1-Distill-Llama-8B_s0.10_channel",
# )

import os
from huggingface_hub import HfApi

def upload_all_models(base_path, user_name):
    # Initialize the Hugging Face API
    api = HfApi()

    # List all directories (models) in the base path
    model_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # Upload each model folder to Hugging Face
    for model_folder in model_folders:
        try:
            # Construct the full path to the model folder
            folder_path = os.path.join(base_path, model_folder)
            # folder_path = model_folder
            # Construct the repository ID
            repo_id = f"{user_name}/{model_folder}"

            # print(folder_path)
            # exit()

            # Perform the upload
            print(f"Uploading {model_folder} to {repo_id}...")
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                repo_type="model"
            )
            print(f"Successfully uploaded {model_folder}.\n")
        except Exception as e:
            print(f"Failed to upload {model_folder}: {e}")

if __name__ == "__main__":
    # Base path where model folders are located
    path_to_models = "./"
    # Replace 'your_username' with your actual Hugging Face username
    huggingface_username = "moon4sake"
    upload_all_models(path_to_models, huggingface_username)