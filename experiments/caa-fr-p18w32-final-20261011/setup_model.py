import argparse
from huggingface_hub import snapshot_download
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);a=p.parse_args()
 snapshot_download('Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',local_dir=a.model_dir)
