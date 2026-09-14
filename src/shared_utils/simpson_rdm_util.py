import logging
import string
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

def load_simpson_rdm(visual_similarity_root: Path) -> np.ndarray:
    visual_similarity_path = (
            visual_similarity_root
            / "simpson_raw.xlsx" )
    visual_similarity_save_path = (
            visual_similarity_root
            / "simpson_2013_visual_rdm.npy")
    visual_similarity_save_excel_path = (
            visual_similarity_root
            / "simpson_2013_visual_rdm.xlsx" )
    if visual_similarity_save_path.exists():
        logger.info(f"加载预处理的 Simpson RDM: {visual_similarity_save_path}")
        return np.load(visual_similarity_save_path)

    if visual_similarity_path.exists():
        logger.info("尝试从原始 Excel 读取...")
        try:
            import pandas as pd
            df = pd.read_excel(visual_similarity_path, sheet_name="List-Upper", index_col=0)
            letters = list(string.ascii_uppercase)
            df = df[df['Letter1'].isin(letters) & df['Letter2'].isin(letters)]
            df_agg = df.groupby(['Letter1', 'Letter2'], as_index=False)['Value'].mean()
            pivot = df_agg.pivot(index='Letter1', columns='Letter2', values='Value')
            pivot = pivot.reindex(index=letters, columns=letters)
            for i in range(len(letters)):
                for j in range(len(letters)):
                    if pd.isna(pivot.iloc[i, j]) and not pd.isna(pivot.iloc[j, i]):
                        pivot.iloc[i, j] = pivot.iloc[j, i]
                    elif pd.isna(pivot.iloc[i, j]) and pd.isna(pivot.iloc[j, i]):
                        pivot.iloc[i, j] = 0
            matrix = pivot.values
            sym = (matrix + matrix.T) / 2
            np.fill_diagonal(sym, 0)
            max_val = sym.max()
            if max_val > 0:
                rdm = 1 - (sym / max_val)
            else:
                rdm = sym
            rdm = (rdm + rdm.T) / 2
            np.fill_diagonal(rdm, 0)
            rdm = np.clip(rdm, 0, 1)
            np.save(visual_similarity_save_path, rdm)
            pd.DataFrame(rdm).to_excel(visual_similarity_save_excel_path, index=False)
            logger.info("从 Excel 读取并转换成功。")
            return rdm
        except Exception as e:
            raise FileNotFoundError(f"无法读取 Simpson Excel: {e}")
    else:
        raise FileNotFoundError(f"未找到{visual_similarity_path}矩阵文件")