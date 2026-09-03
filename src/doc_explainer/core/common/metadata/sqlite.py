import uuid
from pydantic import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, PipelineRun, StepRun, ArtifactCache, RunStatus, StepStatus
from .utils import utcnow_naive

class SQLiteMetadataStore:
    def __init__(self, db_path: str = "metadata.db"):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_pipeline_run(self, pipeline_name: str) -> str:
        session = self.Session()
        run = PipelineRun(
            id=str(uuid.uuid4()),
            pipeline_name=pipeline_name,
            status=RunStatus.PENDING
        )
        session.add(run)
        session.commit()
        run_id = run.id          # capture before closing
        session.close()
        return run_id            # return the ID, not the object

    def update_pipeline_run(self, run_id: str, status: RunStatus):
        session = self.Session()
        run = session.query(PipelineRun).filter_by(id=run_id).first()
        if run:
            run.status = status
            if status in (RunStatus.SUCCESS, RunStatus.FAILED):
                run.finished_at = utcnow_naive()
            session.commit()
        session.close()

    def create_step_run(self, pipeline_run_id: str, node_id: str, step_name: str):
        session = self.Session()
        step_run = StepRun(
            id=node_id,
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            status=StepStatus.PENDING
        )
        session.add(step_run)
        session.commit()
        session.close()

    def update_step_run(self, node_id: str, status: StepStatus, artifact_id: str = None, error: str = None):
        session = self.Session()
        step_run = session.query(StepRun).filter_by(id=node_id).first()
        if step_run:
            step_run.status = status
            if status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.CACHED):
                step_run.finished_at = utcnow_naive()
                if step_run.started_at:
                    step_run.duration = (step_run.finished_at - step_run.started_at).total_seconds()
            if artifact_id:
                step_run.artifact_id = artifact_id
            if error:
                step_run.error = error
            session.commit()
        session.close()

    def get_failed_step_runs(self, run_id: str):
        session = self.Session()
        failed = session.query(StepRun).filter_by(pipeline_run_id=run_id, status=StepStatus.FAILED).all()
        session.close()
        return failed

    def save_cache_entry(self, cache_key: str, artifact_ref):
        import json
        from ..artifacts.artifact import ArtifactRef
        session = self.Session()

        # delete old entry if exists
        session.query(ArtifactCache).filter_by(cache_key=cache_key).delete()
        ref_dict = {
            'id': artifact_ref.id,
            'uri': artifact_ref.uri,
            'type': artifact_ref.type,
            'metadata': artifact_ref.metadata
        }
        entry = ArtifactCache(
            id=str(uuid.uuid4()),
            cache_key=cache_key,
            artifact_ref_json=json.dumps(ref_dict)
        )
        session.add(entry)
        session.commit()
        session.close()

    def get_cached_artifact(self, cache_key: str):
        import json
        from ..artifacts.artifact import ArtifactRef
        session = self.Session()
        entry = session.query(ArtifactCache).filter_by(cache_key=cache_key).first()
        session.close()
        if entry:
            ref_dict = json.loads(entry.artifact_ref_json)
            return ArtifactRef(**ref_dict)
        return None