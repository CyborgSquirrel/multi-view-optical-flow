from util import osp

IN_COLAB = False
try:
  import google.colab
  IN_COLAB = True
except ModuleNotFoundError:
  pass

DRIVE_PATH = "/content/drive"
DRIVE_FILES = osp.join(DRIVE_PATH, "MyDrive")
PROJECT_PATH = osp.join(DRIVE_FILES, "adl4cv")
