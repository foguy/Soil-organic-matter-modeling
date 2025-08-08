import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)


from src.prepare_dataset import DataPreprocessor

