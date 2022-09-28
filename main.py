from models.fsrcnn import FSRCNN
import sys
import argparse
import time
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, help="video frame width", default=144)
    parser.add_argument("-t", "--height", type=int, help="video frame height", default=256)
    parser.add_argument("-c", "--channel", type=int, help="video frame channel", default=1)
    parser.add_argument("-l", "--length", type=int, help="video length(s)", default=5)
    parser.add_argument("-f", "--fps", type=int, help="video frame rate", default=24)
    parser.add_argument("-d", "--device", type=str, help="execute device \"cpu\" or \"gpu\"", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("-s", "--scale_factor", type=int, help="scale factor", default=4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=0)
    args = parser.parse_args()

    width = args.width
    height = args.height
    length = args.length
    channel = args.channel
    fps = args.fps
    device = args.device
    batch_size = args.batch_size
    scale_factor = args.scale_factor
    if device == "gpu" and not torch.cuda.is_available():
        print("\n!!! GPU is not available for this computer !!!\n", file=sys.stderr)
    device = torch.device("cuda:0" if torch.cuda.is_available() and device=="gpu" else "cpu")
    model = FSRCNN(scale_factor=scale_factor).to(device)

    video_frames = np.random.rand(fps*length, channel, height, width)
    tensor_frames = torch.from_numpy(video_frames).to(device).float()
    
    start = time.perf_counter()
    if batch_size == 0:
        model(tensor_frames)
    else:
        for idx in range(0, fps*length, batch_size):
            model(tensor_frames[idx:idx+batch_size,:,:,:])
    print("{}s".format(time.perf_counter() - start))