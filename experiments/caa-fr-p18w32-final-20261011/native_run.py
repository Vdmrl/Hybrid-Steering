"""Single GPU BF16 native port; same frozen additive hooks and decoding."""
import argparse,os,uuid,hashlib,json,subprocess,time
from pathlib import Path
def main():
 import torch
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 import run
 from search import execute
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.runtime_mode='single-native';a.config=a.config.resolve();a.output=a.output.resolve()
 assert torch.cuda.device_count()==1
 torch.manual_seed(20260911);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 device=torch.device('cuda',0);torch.cuda.set_device(0)
 os.environ['STEERING_LOAD_SESSION']=uuid.uuid4().hex
 os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','0')
 os.environ['STEERING_GPU_UUIDS']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','--id='+os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip()
 model_path=os.environ.get('STEERING_MODEL_PATH','/workspace/models/Qwen3.5-9B')
 started=time.monotonic()
 model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0}).eval()
 assert all(t.device.type=='cuda' for t in list(model.parameters())+list(model.buffers()))
 tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 original=run.load_colleague(Path(__file__).with_name('colleague.py'))
 print('NATIVE_BF16_LOADED',json.dumps({'load_seconds':time.monotonic()-started,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'revision':original.REVISION,'allocated_mib':torch.cuda.memory_allocated()/2**20,'offload':False}),flush=True)
 execute(model,tokenizer,device,original,a,run.generate)
if __name__=='__main__':main()
