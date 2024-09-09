# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/9 15:10


import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# 导入DESeq2包
deseq = importr("DESeq2")

# 导入表达矩阵和实验设计表
counts = pd.read_csv("counts.csv", index_col=0)
design = pd.read_csv("design.csv", index_col=0)

# 转换为R数据框
with localconverter(robjects.default_converter + pandas2ri.converter):
    counts_r = robjects.conversion.py2rpy(counts)
    design_r = robjects.conversion.py2rpy(design)

# 创建DESeqDataSet对象
dds = deseq.DESeqDataSetFromMatrix(countData=counts_r, colData=design_r, design=~condition)

# 进行标准化和差异分析
dds = deseq.DESeq(dds)
res = deseq.results(dds)

# 筛选差异基因
res = pd.DataFrame(pandas2ri.ri2py(res))
res_sig = res.loc[(res['padj'] < 0.05)(abs(res['log2FoldChange']) > 1)]

# 保存结果
res_sig.to_csv("deseq2_results.csv")
