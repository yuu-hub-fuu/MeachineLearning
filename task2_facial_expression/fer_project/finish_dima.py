"""Finish the authorized Dima run: adaptation, diagnostics, reload, packaging."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent

def alive(pid):
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(handle)
    return code.value==259

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int,required=True);args=p.parse_args()
    deadline=time.monotonic()+3*3600
    while not (OUT/'dima_finetune_results.json').exists():
        if not alive(args.training_pid):raise RuntimeError('Training stopped without final results; inspect dima_finetune.err')
        if time.monotonic()>deadline:raise TimeoutError('Training did not complete within three hours')
        time.sleep(5)
    assert json.loads((OUT/'dima_finetune_results.json').read_text())['completed']
    assert json.loads((OUT/'dima_tta_verification.json').read_text())['reset_verified']
    for name in ['tta_dima.py','analyze_dima.py','analyze_dima_features.py','infer_dima_finetuned.py','summarize.py']:
        print(json.dumps({'event':'stage_start','script':name}),flush=True)
        subprocess.run([sys.executable,'-u','-X','utf8',str(HERE/name)],check=True)
    print(json.dumps({'event':'dima_delivery_completed'}),flush=True)

if __name__=='__main__':main()
