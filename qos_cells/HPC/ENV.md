# ENV — node used for E8–E11

Captured 2026-11-05 18:45:16 UTC on `quant1`.

| item | value |
|---|---|
| host | quant1 |
| CPU cores (nproc) | 128 |
| CPU model | AMD EPYC 7542 32-Core Processor |
| RAM | 2003 GB total |
| GPUs | 8 NVIDIA RTX A6000 (48 GB each) |
| NVIDIA driver / CUDA | 575.57.08 / 12.9 |
| SLURM | yes (sbatch present; partitions Normal: quant2, Long: quant1 = this node, Devel: quant3 drained). Not used: jobs run directly here with nohup. |
| internet | yes (pypi.org and github.io return HTTP 200) |
| system Python | Python 3.12.3 (untouched; venv in HPC/.venv) |
| disk (project, beegfs) | 8,7T free of 24T |
| disk (/) | 40G free of 216G |

## nvidia-smi
```
index, name, memory.total [MiB], driver_version
0, NVIDIA RTX A6000, 49140 MiB, 575.57.08
1, NVIDIA RTX A6000, 49140 MiB, 575.57.08
2, NVIDIA RTX A6000, 49140 MiB, 575.57.08
3, NVIDIA RTX A6000, 49140 MiB, 575.57.08
4, NVIDIA RTX A6000, 49140 MiB, 575.57.08
5, NVIDIA RTX A6000, 49140 MiB, 575.57.08
6, NVIDIA RTX A6000, 49140 MiB, 575.57.08
7, NVIDIA RTX A6000, 49140 MiB, 575.57.08
```

## free -g
```
               total        used        free      shared  buff/cache   available
Mem:            2003          17        1990           0           5        1986
Swap:              7           0           7
```

Note: the brief says 1 GPU; the node has 8× A6000. Jobs are spread over GPUs 0–4 (see PROGRESS.md).
