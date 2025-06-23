from dataclasses import dataclass

# we're using dataclasses for empty class
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str