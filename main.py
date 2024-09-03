import sys
import os
import shap
from fastapi.responses import JSONResponse


# Add the project root directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import make_inference, model, test_df
from src.visualization import explain_result
from fastapi.responses import StreamingResponse

# from src.p

from fastapi import FastAPI

app = FastAPI()

@app.get("/predict/{lead_id}")
async def predict(lead_id: str):
    prediction = make_inference(lead_id)
    return prediction

@app.get("/visualize/{lead_id}")
async def visualize(lead_id: str):
    # Ensure to load or preprocess data as required
    explainer = shap.Explainer(model)
    
    # Get visualization results
    visualizations = explain_result(test_df, explainer, lead_id)

    return visualizations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
