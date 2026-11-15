"""Finish the already running queue, then optimize, verify, and package outputs.

For a fresh complete run, omit --wait-for-pid. A Windows process handle avoids
confusing a recycled PID with the original training process.
"""
import argparse, ctypes, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('FER_ROOT', r'D:\FER2026task2'))
OUT = HERE.parent


def run(script, *arguments):
    log = ROOT/'runs'/f'{Path(script).stem}.log'
    print(json.dumps({'event': 'launch', 'script': script, 'log': str(log)}), flush=True)
    with log.open('w', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-u', '-X', 'utf8', str(HERE/script), *arguments],
                       cwd=HERE, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wait-for-pid', type=int)
    args = p.parse_args()
    status = ROOT/'runs'/'finish_pipeline_status.json'

    def update(stage, **extra):
        obj = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **extra}
        status.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(obj), flush=True)

    try:
        if args.wait_for_pid:
            if os.name != 'nt': raise RuntimeError('--wait-for-pid is Windows-only')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.WaitForSingleObject.restype = ctypes.c_uint32
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x00100000, False, args.wait_for_pid)
            if handle:
                update('waiting_for_core_training', pid=args.wait_for_pid)
                try:
                    while True:
                        state = kernel.WaitForSingleObject(handle, 10000)
                        if state == 0: break
                        if state != 258: raise ctypes.WinError(ctypes.get_last_error())
                finally: kernel.CloseHandle(handle)
        else:
            update('core_training'); run('run_experiments.py', '--epochs', '12')
        selection = json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
        if not selection.get('completed'): raise RuntimeError('Core queue did not finish successfully')
        warnings=[]
        for stage,script,arguments in [
            ('tta_features_and_error_analysis','optimize_predictions.py',[]),
            ('pretrained_feature_adaptation','upgrade_experiments.py',['--wait-weights-seconds','3600'])]:
            update(stage)
            try:run(script,*arguments)
            except subprocess.CalledProcessError as error:
                warnings.append({'stage':stage,'error':repr(error),'log':str(ROOT/'runs'/f'{Path(script).stem}.log')})
                print(json.dumps({'event':'optional_stage_failed',**warnings[-1]}),flush=True)
        (OUT/'pipeline_warnings.json').write_text(json.dumps(warnings,ensure_ascii=False,indent=2),encoding='utf-8')
        update('verifying_and_packaging'); run('summarize.py')
        update('completed_with_warnings' if warnings else 'completed', warnings=warnings, submission=str(OUT/'submission.csv'), archive=str(OUT/'FER2026_source_and_results.zip'))
    except Exception as error:
        update('failed', error=repr(error)); raise


if __name__ == '__main__': main()
