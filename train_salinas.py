import subprocess
import time
import itertools

# 所有超参数组合
weight_noise_list = list(range(50, 51, 10))                     # 10 to 100, step 10
weight_rep_list = list(range(0, 301, 5))                         # 0 to 100, step 2
param_combinations = list(itertools.product(weight_noise_list, weight_rep_list))

# 单 GPU ID
gpu_id = 0
device = f"cuda:{gpu_id}"

total_jobs = len(param_combinations)
print(f"Total {total_jobs} jobs will be run on GPU {device}")

if __name__ == '__main__':
    for idx, (weight_noise, weight_rep) in enumerate(param_combinations, 1):
        cmd = [
            "python", "train.py",
            f"--weight_representation={weight_rep}",
            f"--weight_noise={weight_noise}",
            f"--device={device}",
            "--log=salinas",
            "--dataset=salinas"
        ]
        print(f"\n[{idx}/{total_jobs}] Running: --weight_representation={weight_rep}, --weight_noise={weight_noise}")
        proc = subprocess.Popen(cmd)
        proc.wait()  # 等待当前进程完成再进入下一轮
        print(f"[{idx}/{total_jobs}] Completed.")
