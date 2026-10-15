clear; clc;
% -------- 修改这里加载你的真实标签、聚类预测标签 --------
addpath('D:\硬盘备份\mega\work\FDPAN-main')
addpath('../UCI','../Complicate','../evaluation','../drawGraph')
load seeds.mat   % 里面变量 y_true, y_pred
load fdpan_predic_seed_label.mat


% ---------------------- 替换为你的真实数据 ----------------------

dataset_name = "Seeds_FDPAN";
% ----------------------------------------------------------------

% ===== 核心：计算列联表 =====
contingency_table = confusionmat(label, cl);

disp('================ Contingency Table 列联表 ================');
disp(contingency_table);

% 获取类别编号
uTrue  = unique(label);
uPred  = unique(cl);

% 包装成带行列名称的table，便于阅读和导出csv
rowNames  = compose("True_%d", uTrue);
colNames  = compose("Pred_%d", uPred);
T = array2table(contingency_table,...
    'RowNames',rowNames,...
    'VariableNames',colNames);

disp(' ');
disp('带行列标识的列联表：');
disp(T);



% 原始列联表
C_orig = T;

% 匈牙利得到最优列置换索引: [3,2,1]
perm_col = [3,2,1];
C_matched = C_orig(:, perm_col);

disp('置换（匹配后）的列联表');
disp(C_matched);

