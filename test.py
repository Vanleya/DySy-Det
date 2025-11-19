import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict

from datasets.dysy_dataset import CustomDataset
from models.dysydet import MyModel
from utils.seed import set_seed
from engine.evaluate import evaluate
from utils.params import get_args_parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dataset_name == "UniversalFakeDetect":
        test_order = [
            "san.json", "deepfake.json", "crn.json", "imle.json", "ldm_200_cfg.json", 
            "seeingdark.json", "stargan.json", "dalle.json", "guided.json", 
            "glide_50_27.json", "stylegan.json", "progan.json", "cyclegan.json", 
            "biggan.json", "gaugan.json", "glide_100_10.json", "glide_100_27.json", 
            "ldm_100.json", "ldm_200.json"
        ]
    else:
        test_order = [
            "BigGAN.json", "ADM.json", "Midjourney.json", "glide.json", 
            "VQDM.json", "stable_diffusion_v_1_4.json", "stable_diffusion_v_1_5.json", 
            "wukong.json"
        ]

    model = MyModel(args.clip_path, args.sd_path, args.num_classes, args.beta, len(args.t), device=device)
    state_dict = torch.load(args.model_ckpt, map_location=device, weights_only=True)
    if all(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    model_name = os.path.splitext(os.path.basename(args.model_ckpt))[0]
    result_file = os.path.join(args.save_path, f"{model_name}.txt")
    os.makedirs(args.save_path, exist_ok=True)
    results = {}
    
    for test_file in test_order:
        test_path = os.path.join(args.test_dataset, test_file)
        dataset = CustomDataset(test_path, args.clip_path, args.img_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc, ap = evaluate(model, dataloader, device, args.t, args.ensemble_size, args.prompt, args.beta)
        results[test_file] = (acc, ap)
        print(f"{test_file}: Acc={acc:.4f}, AP={ap:.4f}")

        with open(result_file, 'a') as f:
            f.write(f"{test_file}\tAcc: {acc:.4f}\tAP: {ap:.4f}\n")

    accs = [v[0] for v in results.values()]
    aps = [v[1] for v in results.values()]
    avg_acc = np.mean(accs)
    avg_ap = np.mean(aps)

    with open(result_file, 'a') as f:
        f.write(f"\nAverage Acc: {avg_acc:.4f}\n")
        f.write(f"Average AP: {avg_ap:.4f}\n")
    print(f"\n[✔] Results saved to: {result_file}")

if __name__ == "__main__":
    main()