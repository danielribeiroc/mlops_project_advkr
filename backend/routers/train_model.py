from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from services.dvc_service import DVCService

router = APIRouter()
dvc_service = DVCService(dataset_dir="./datasets")  # Initialize the service

@router.post("/train-model")
async def train_model(files: List[UploadFile] = File(...)):
    print("Received files for training.")
    # Save files and track with DVC
    try:
        print("Saving files...")
        saved_files = dvc_service.save_files(files)  # Save files locally
        print("Tracking files with DVC...")
        dvc_service.track_with_dvc(saved_files)  # Track with DVC and push
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "Files received, tracked with DVC, and pushed to remote storage."}
