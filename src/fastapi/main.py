import logging
import os

from utils.get_data import get_data
import uvicorn
from utils.predict import predict_text
from utils.preprocess_data import preprocess_data
from utils.train import train

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

# logger
logger = logging.getLogger(__name__)

# FastAPI
tags_metadata = [
    {"name": "Model Training", "description": "ETL and Model training pipelines"},
    {"name": "Inference", "description": "Model Inference utils"},
]
app = FastAPI(
    title="Task5",
    description="Task 5 for EPAM ML OPS Course",
    openapi_tags=tags_metadata,
)

# Redirect to swagger
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """
    Redirect to Swagger documentation when accessing the root path
    """
    return RedirectResponse("/docs")


@app.post("/train_model/", tags=["Model Training"])
async def train_pipeline(
    background_tasks: BackgroundTasks,
    model_name: str = Query(
        "Random Forest",
        enum=["RandomForest", "LogisticRegression", "SVC"],
        description="Select the model",
    ),
):
    # Create directories if needed
    if not os.path.exists("data/models/"):
        os.makedirs("data/models/", exist_ok=True)

    # Download the data if needed
    if not os.path.exists("data/jigsaw-toxic-comment-train.csv"):
        get_data()

    # Preprocess the data if needed
    if not os.path.exists("data/preprocessed_data.csv"):
        preprocess_data()

    # Train the model
    if os.path.exists(f"data/models/{model_name}.joblib"):
        os.remove(f"data/models/{model_name}.joblib")
    train(model_name)

    # Success if everything ok
    return {"Status":"Done"}


@app.get("/is_model_trained/", tags=["Model Training"])
async def is_model_trained(
    model_name: str = Query(
        "Random Forest",
        enum=["RandomForest", "LogisticRegression", "SVC"],
        description="Select the model",
    ),
):
    if os.path.exists(f"data/models/{model_name}.joblib"):
        return {"Status": "Model trained"}
    else:
        return {"Status": "Model not trained"}


@app.post("/predict/", tags=["Inference"])
async def predict(
    model_name: str = Query(
        "Random Forest",
        enum=["RandomForest", "LogisticRegression", "SVC"],
        description="Select the model",
    ),
    text: list = Query(default=None, description="Write the text"),
):
    # check for None
    if text is None:
        raise HTTPException(status_code=404, detail="Please, add some text")

    # vectorize data
    if not os.path.exists(f"data/models/vectorizer.joblib"):
        raise HTTPException(status_code=404, detail="Vectorizer not found")

    # load model
    if not os.path.exists(f"data/models/{model_name}.joblib"):
        raise HTTPException(status_code=404, detail="Model not found")

    res = predict_text(model_name, text)

    return {t: str(res[idx]) for idx, t in enumerate(text)}


@app.get("/healthcheck/")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090, reload=True)

