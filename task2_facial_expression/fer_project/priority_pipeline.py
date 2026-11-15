"""Finish the current ablation, then prioritize task-finetuned checkpoints."""
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
from finish_pipeline import ROOT,OUT,HERE,run

def status(stage,**kwargs):
    record={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs}
    (ROOT/'runs'/'finish_pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)

def main():
    pid=int(sys.argv[1]);status('waiting_for_current_scn',pid=pid)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_bool,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(handle,10000)==258:pass
        finally:kernel.CloseHandle(handle)
    variants=['baseline','local']
    if (ROOT/'runs'/'scn_s42_w0'/'completed.json').exists():variants.append('scn')
    status('preserving_completed_baselines',variants=variants)
    run('run_experiments.py','--epochs','12','--variants',*variants,'--no-refit')
    status('priority_pretrained_feature_adaptation')
    run('upgrade_experiments.py','--kinds','hf_dima','rmn','qcs','hf_mo','hf_trpakov','nla','--wait-weights-seconds','3600')
    selected=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    if not selected.get('upgrade',{}).get('adopted_over_core'):
        status('fallback_full_refit')
        run('run_experiments.py','--epochs','12','--variants',*variants)
        status('fallback_tta');run('optimize_predictions.py')
    status('verifying_and_packaging');run('summarize.py')
    status('completed',submission=str(OUT/'submission.csv'))

if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=repr(error));raise
