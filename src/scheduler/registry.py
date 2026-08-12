# src/scheduler/registry.py
from pathlib import Path
from typing import Dict, Any

def get_pipeline_registry(src_dir: Path, data_dir: Path, results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    子工程注册表（唯一事实来源）
    所有路径由 run.py 计算后传入，避免硬编码。
    """
    return {
        "preprocess": {
            "name": "数据预处理",
            "script": src_dir / "pipeline1_empirical_rsa" / "run_preprocess.py",
            "requires_subject": True,
            "depends_on": [],
            "done_file": lambda subj: data_dir / "derivatives" / f"sub-{subj}_coding_epo.fif"
        },
        "empirical": {
            "name": "工程一：动态RSA",
            "script": src_dir / "pipeline1_empirical_rsa" / "run_rsa_analysis.py",
            "requires_subject": True,
            "depends_on": ["preprocess"],
            "done_file": lambda subj: results_dir / "stats" / f"sub-{subj}_rsa_results.npy"
        },
        "group_stats": {
            "name": "组水平统计",
            "script": src_dir / "pipeline1_empirical_rsa" / "run_group_stats.py",
            "requires_subject": False,
            "depends_on": ["empirical"],
            "done_file": lambda _: results_dir / "stats" / "group_p_values.npy"
        },
        "simulation": {
            "name": "工程二：模拟验证",
            "script": src_dir / "pipeline2_simulation_validate" / "run_simulation.py",
            "requires_subject": False,
            "depends_on": [],
            "done_file": lambda _: results_dir / "simulation_sweep.png"
        },
        "rnn": {
            "name": "工程三：RNN建模",
            "script": src_dir / "pipeline3_rnn_modeling" / "run_rnn.py",
            "requires_subject": False,
            "depends_on": [],
            "done_file": lambda _: results_dir / "models" / "ctrnn_seq_memory.pt"
        }
    }