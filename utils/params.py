import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--clip_path', type=str, default="")
    parser.add_argument('--sd_path', type=str, default="")


    parser.add_argument('--train_dataset', type=str, default="./train_dataset/train.json")
    parser.add_argument('--test_dataset', type=str, default="./test_dataset/val")

    parser.add_argument('--img_size', nargs='+', type=int, default=(512, 512))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ensemble_size', type=int, default=4)
    parser.add_argument('--t', nargs='+', type=int, default=[200, 250, 300])
    parser.add_argument('--prompt', default='', type=str)

    parser.add_argument('--save_path', type=str, default="./checkpoints/weights")
    parser.add_argument('--log_path', type=str, default="./Here/log.txt")

    parser.add_argument('--model_ckpt', type=str, required=True)

    parser.add_argument('--dataset_name', type=str, default="GenImage", choices=["GenImage", "UniversalFakeDetect"])
    parser.add_argument('--beta', type=float, default=0.99)
    return parser