from sqlalchemy import Column, String, Float, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import declarative_base
from enum import Enum as PyEnum
from .utils import utcnow_naive

Base = declarative_base()


class RunStatus(PyEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'


class StepStatus(PyEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'
    CACHED = 'cached'


class PipelineRun(Base):
    __tablename__ = 'pipeline_runs'
    id = Column(String, primary_key=True)
    pipeline_name = Column(String, nullable=False)
    status = Column(Enum(RunStatus), default=RunStatus.PENDING)
    started_at = Column(DateTime(timezone=True), default=utcnow_naive)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    run_metadata = Column('metadata', Text, nullable=True)   # attribute name is run_metadata, column name is metadata


class StepRun(Base):
    __tablename__ = 'step_runs'
    id = Column(String, primary_key=True)  # node id
    pipeline_run_id = Column(String, ForeignKey('pipeline_runs.id'))
    step_name = Column(String, nullable=False)
    status = Column(Enum(StepStatus), default=StepStatus.PENDING)
    started_at = Column(DateTime(timezone=True), default=utcnow_naive)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Float, nullable=True)  # seconds
    artifact_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)


class ArtifactCache(Base):
    __tablename__ = 'artifact_cache'
    id = Column(String, primary_key=True)
    cache_key = Column(String, unique=True, nullable=False)
    artifact_ref_json = Column(Text, nullable=False)   # stores the whole ArtifactRef as JSON
    created_at = Column(DateTime(timezone=True), default=utcnow_naive)

