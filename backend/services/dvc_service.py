import os
from typing import List
from fastapi import HTTPException, UploadFile
from dvc.repo import Repo

class DVCService:
    def __init__(self, dataset_dir: str = "./datasets"):
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)  # Ensure the dataset folder exists
        self.repo = Repo()  # Initialize the DVC repo

    def save_files(self, files: List[UploadFile]) -> List[str]:
        """Save uploaded files to the dataset directory."""
        saved_files = []
        for file in files:
            if file.content_type != "text/plain":
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload .txt files only.")
            
            file_path = os.path.join(self.dataset_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())  # Save file content
            saved_files.append(file_path)
        
        return saved_files

    def track_with_dvc(self, files: List[str]):
        """Track the saved files with DVC and push to remote."""
        try:
            print("Testing DVC")
            for file_path in files:
                self.repo.add(file_path)
                print(f"Tracked {file_path} with DVC.")

            self.repo.scm.add(["."])
            self.repo.scm.commit("Add training data via API")

            #self.repo.push()
            #self.repo.scm.push()
            print("Files and metadata pushed to remote storage.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during DVC/Git operation: {str(e)}")
