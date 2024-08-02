"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import torch
import torch.distributed as dist

import os
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

print(local_rank)