import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    # TODO: Add more options if needed
    parser.add_argument(
        '--model',
        default='RCA',
        type=str,
        help='Model to use'
    )
    parser.add_argument(
        '--use_cuda',
        default=False,
        action='store_true',
        help='Use GPU'
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='GPU id to use'
    )
    parser.add_argument(
        '--data_dir',
        default='./data',
        type=str,
        help='Directory path to the dataset'
    )
    parser.add_argument(
        '--num_frames',
        default=20,
        type=int,
        help='Number of frames per video'
    )
    parser.add_argument(
        '--num_levels',
        default=8,
        type=int,
        help='Number of levels'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of workers'
    )
    parser.add_argument(
        '--lr_rate',
        default=1e-3,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-5,
        type=float,
        help='Weight decay'
    )
    parser.add_argument(
        '--n_epochs',
        default=20,
        type=int,
        help='Number of epochs'
    )
    parser.add_argument(
        '--resume_path',
        default=None,
        type=str,
        help='Path to the checkpoint'
    )
    parser.add_argument(
        '--log_interval',
        default=10,
        type=int,
        help='Log interval'
    )
    parser.add_argument(
        '--record_interval',
        default=2,
        type=int,
        help='Record interval'
    )
    parser.add_argument(
        '--checkpoint_interval',
        default=20,
        type=int,
        help='Checkpoint interval'
    )
    args = parser.parse_args()
    return args
    