import matplotlib.pyplot as plt
from python.evaluation.evaluate import evaluate
if __name__=='__main__':
    d=evaluate(); plt.bar(d.keys(),[v['mean_error'] for v in d.values()]); plt.ylabel('reconstruction error'); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig('models/error_summary.png',dpi=140)
