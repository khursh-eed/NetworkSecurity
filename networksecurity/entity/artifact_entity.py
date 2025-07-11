from dataclasses import dataclass

# we're using dataclasses for empty class
# output of dataingestion
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


#  output fromdata validation
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_Score: float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    model_trainer_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
