import unittest
from unittest.mock import patch
import argparse
from homework.train import *
import sys

def parse_args(args):
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    return parser.parse_args(args)

class train_test(unittest.TestCase):

    @patch('sys.argv', ['train.py', '--log_dir', 'logs', '-n', '20', '-lr', '0.01', '-g', '0.9'])
    def test_train_random_configuration(self):
        args = parse_args(sys.argv[1:])
        train(args,  data_folder='../drive_data')




    def test_check_model_size(self):
        model = Planner()

        total_params = 0
        for param in model.parameters():
            total_params += param.numel() * param.element_size()
        return total_params
