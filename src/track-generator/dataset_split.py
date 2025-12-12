from enum import StrEnum


class DatasetSplit(StrEnum):
  FULL = "full"
  TRAIN = "train"
  VALIDATION = "validation"
  TEST = "test"
