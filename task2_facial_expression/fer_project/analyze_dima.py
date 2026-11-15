"""Compare selected validation predictions with the frozen-head baseline."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
OUT=Path(__file__).resolve().parent.parent
ROOT=Path(r'D:\FER2026task2')
NAMES=['Angry','Fear','Happy','Sad','Surprise','Neutral']

def main():
    result=json.loads((OUT/'dima_finetune_results.json').read_text())
    final=np.load(OUT/('dima_final_validation.npz' if (OUT/'dima_final_validation.npz').exists() else 'dima_finetune_validation.npz'))
    baseline=np.load(ROOT/'runs'/'upgrade_hf_dima_linear_holdout_s42'/'validation_probabilities.npz')
    assert np.array_equal(final['indices'],baseline['indices']) and np.array_equal(final['labels'],baseline['labels'])
    y=final['labels'];a=baseline['probabilities'].argmax(1);b=final['probabilities'].argmax(1)
    ca=confusion_matrix(y,a,labels=range(6));cb=confusion_matrix(y,b,labels=range(6))
    r={'baseline_accuracy':float((a==y).mean()),'selected_accuracy':float((b==y).mean()),
       'corrected_errors':int(((a!=y)&(b==y)).sum()),'new_errors':int(((a==y)&(b!=y)).sum()),
       'selected_classification_report':classification_report(y,b,target_names=NAMES,output_dict=True,zero_division=0),
       'baseline_confusion':ca.tolist(),'selected_confusion':cb.tolist(),
       'interpretation':'Descriptive diagnostics on model-selection holdout; no independent significance claim.'}
    (OUT/'dima_error_comparison.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    lines=['# Dima 训练后的错误变化','','与冻结主干线性适配（82.35%）比较。此处为参与模型选择的留出验证，存在外部预训练重叠风险。','',
           f"纠正原有错误 {r['corrected_errors']} 张，新增错误 {r['new_errors']} 张。",'',
           '|类别|样本数|原召回率|最终召回率|变化（百分点）|','|---|---:|---:|---:|---:|']
    for i,n in enumerate(NAMES):
        count=int(ca[i].sum());ra=ca[i,i]/count;rb=cb[i,i]/count
        lines.append(f'|{n}|{count}|{ra:.2%}|{rb:.2%}|{100*(rb-ra):+.2f}|')
    lines+=['','|真实→错误预测|原数量|最终数量|','|---|---:|---:|']
    pairs=sorted([(int(ca[i,j]),i,j) for i in range(6) for j in range(6) if i!=j],reverse=True)[:8]
    for _,i,j in pairs:lines.append(f'|{NAMES[i]}→{NAMES[j]}|{ca[i,j]}|{cb[i,j]}|')
    lines+=['','训练没有依据这些验证错误改标签；原始训练标签始终保留。完整误分类样本见 `dima_validation_errors.csv`。']
    (OUT/'Dima错误分析.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:r[k] for k in ['baseline_accuracy','selected_accuracy','corrected_errors','new_errors']}))

if __name__=='__main__':main()
