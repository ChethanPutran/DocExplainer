import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # ensures handlers are set even if already configured
)

from doc_explainer.orchestrator import step, pipeline
from doc_explainer.orchestrator.artifacts.local import LocalArtifactStore
from doc_explainer.orchestrator.metadata.models import StepRun
from doc_explainer.orchestrator.metadata.sqlite import SQLiteMetadataStore



@step(annotations={"description": "Load the data"})
def load_data():
    return {"x": 1, "y": 2}

@step(annotations={"description": "Preprocess the data"})
def preprocess(data):
    data["x"] = data["x"] * 2
    return data

@step(annotations={"description": "Train the model"})
def train(data):
    return {"model": "trained", "data": data}

@pipeline
def my_pipeline():
    data = load_data()
    processed = preprocess(data)
    model = train(processed)
    return model

def test_pipeline_run(capsys):   # capsys captures stdout
    run_id = my_pipeline.run()
    print(f"Run ID: {run_id}")