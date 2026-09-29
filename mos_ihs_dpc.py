import numpy as np
import pickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

# ================== MOS: Mean-shift Outlier Score (MOD论文原版，同轮统一更新) ==================
def mos_mean_shift_outlier_score(X, k=3, tune_round=3):
    """
    Mean-Shift Outlier Detector MOD
    :param X: (n,d)原始数据
    :param k: k近邻
    :param tune_round:迭代轮数，论文默认3
    :return: outlier_scores 每个点位移得分，越大越可能是噪声
    """
    n, dim = X.shape
    X_current = X.copy()

    for _ in range(tune_round):
        X_new = X_current.copy() # 论文版本，一轮全部基于本轮起点，统一更新，不是inplace
        dist_mat = pairwise_distances(X_current)
        for i in range(n):
            dist_i = dist_mat[i, :]
            # 排除自己，选最近k个邻居
            idx_neigh = np.argsort(dist_i)[1:k+1]
            neigh_points = X_current[idx_neigh, :]
            X_new[i] = np.mean(neigh_points, axis=0)
        X_current = X_new

    # MOD得分：原始位置vs平滑后位置欧式距离
    outlier_scores = np.linalg.norm(X - X_current, axis=1)
    return outlier_scores

# -------------------------- 新增：聚类精度 ACC 函数 --------------------------
def clustering_acc(y_true, y_pred):
    """
    Clustering Accuracy，Hungarian bestMap
    :param y_true: 真实标签
    :param y_pred: 聚类预测标签
    :return: acc 0~1
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    true_unique = np.unique(y_true)
    pred_unique = np.unique(y_pred)
    n_t = len(true_unique)
    n_p = len(pred_unique)

    cost = np.zeros((n_t, n_p), dtype=int)
    for i, t in enumerate(true_unique):
        for j, p in enumerate(pred_unique):
            cost[i, j] = np.sum((y_true == t) & (y_pred == p))

    row_idx, col_idx = linear_sum_assignment(-cost) #  Hungarian bestMap
    correct = cost[row_idx, col_idx].sum()
    return round(correct / len(y_true), 4)

# -------------------------- 数据集工具函数【完全保留原样，输入导入不变】 --------------------------
def load_dataset(path):
    """加载pkl数据集"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"数据集文件不存在: {path}")
        return None
    except Exception as e:
        print(f"读取数据集异常 {path}: {e}")
        return None

# 各数据集完整超参配置【原样保留】
dataset_params = {
    "iris": {"dc_ratio": 0.03, "init_lam": 0.15},
    "ecoli": {"dc_ratio": 0.028, "init_lam": 0.16},
    "thyroid": {"dc_ratio": 0.022, "init_lam": 0.17},
    "seeds": {"dc_ratio": 0.032, "init_lam": 0.14},
    "zoo": {"dc_ratio": 0.04, "init_lam": 0.12},
    "wine": {"dc_ratio": 0.026, "init_lam": 0.19},
    "mnist2d": {"dc_ratio": 0.02, "init_lam": 0.2},
    "usps2d": {"dc_ratio": 0.025, "init_lam": 0.18},
    "data_PenDigits2d": {"dc_ratio": 0.023, "init_lam": 0.19},
    "MSRA25": {"dc_ratio": 0.021, "init_lam": 0.21},
    "Palmdata": {"dc_ratio": 0.024, "init_lam": 0.20},
    "spambase": {"dc_ratio": 0.018, "init_lam": 0.22},
    "msplice": {"dc_ratio": 0.020, "init_lam": 0.21},
    'oliver100': {"dc_ratio": 0.9999, "init_lam": 8},
    "Aggregation": {"dc_ratio": 0.025, "init_lam": 0.16},
    "R15": {"dc_ratio": 0.020, "init_lam": 0.13},
    "Pathbased": {"dc_ratio": 0.024, "init_lam": 0.17},
    "Compound": {"dc_ratio": 0.027, "init_lam": 0.18},
    "Jain": {"dc_ratio": 0.022, "init_lam": 0.15},
    "a3": {"dc_ratio": 0.01, "init_lam": 0.14},
    "s4": {"dc_ratio": 0.01, "init_lam": 0.18},
    "unbalance": {"dc_ratio": 0.01, "init_lam": 0.01}
}

# -------------------------- 迭代硬筛选密度峰值聚类 IHS-DPC【完全原样保留】 --------------------------
class IterHardScreenDPC:
    def __init__(self, dc_ratio=0.02, init_lam=0.1):
        self.dc_ratio = dc_ratio
        self.init_lam = init_lam
        # 全局固定参数
        self.lam_step = 0.1
        self.max_iter = 4

        self.X = None
        self.n = 0
        self.dc = 0
        self.v = None
        self.rho = None
        self.delta = None
        self.gamma = None
        self.labels = None
        self.center_idx = []

    def _calc_dc(self, X):
        n = X.shape[0]
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(np.linalg.norm(X[i] - X[j]))
        dists = np.sort(dists)
        pos = int(len(dists) * self.dc_ratio)
        return dists[pos]

    def _weighted_rho(self, X, v, dc):
        n = X.shape[0]
        rho = np.zeros(n)
        for i in range(n):
            xi = X[i]
            for j in range(n):
                if i == j or v[j] < 1e-6:
                    continue
                dist = np.linalg.norm(xi - X[j])
                rho[i] += v[j] * np.exp(- (dist ** 2) / (dc ** 2))
        return rho

    def _update_v(self, rho, lam):
        loss = rho
        n = len(rho)
        v = np.zeros(n)
        for i in range(n):
            v[i] = 1.0 if loss[i] > lam else 0.0
        return v

    def _calc_delta_gamma(self, rho, X):
        n = X.shape[0]
        delta = np.full(n, np.inf)
        for i in range(n):
            for j in range(n):
                if rho[j] > rho[i]:
                    dist = np.linalg.norm(X[i] - X[j])
                    if dist < delta[i]:
                        delta[i] = dist
        max_delta = np.max(delta[np.isfinite(delta)])
        delta[np.isinf(delta)] = max_delta
        gamma = rho * delta
        return delta, gamma

    def _assign_label(self, rho, delta, gamma, X, true_k):
        n = X.shape[0]
        sort_idx = np.argsort(-gamma)
        center_num = true_k
        self.center_idx = sort_idx[:center_num]

        labels = np.full(n, -1)
        for c in self.center_idx:
            labels[c] = c

        ord_idx = np.argsort(-rho)
        for i in ord_idx:
            if labels[i] != -1:
                continue
            xi = X[i]
            min_dist = np.inf
            best_label = -1
            for j in range(n):
                if labels[j] == -1 or rho[j] <= rho[i]:
                    continue
                d = np.linalg.norm(xi - X[j])
                if d < min_dist:
                    min_dist = d
                    best_label = labels[j]
            labels[i] = best_label

        unique_centers = list(np.unique(labels[labels != -1]))
        map_dict = {old: idx for idx, old in enumerate(unique_centers)}
        new_labels = np.array([map_dict[x] if x != -1 else -1 for x in labels])
        return new_labels

    def fit(self, X, true_cluster_num):
        # Z‑score标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X = X
        self.n = X.shape[0]
        self.dc = self._calc_dc(X)
        self.v = np.ones(self.n)
        lam = self.init_lam

        for t in range(self.max_iter):
            self.rho = self._weighted_rho(X, self.v, self.dc)
            self.v = self._update_v(self.rho, lam)
            self.delta, self.gamma = self._calc_delta_gamma(self.rho, X)
            self.labels = self._assign_label(self.rho, self.delta, self.gamma, X, true_cluster_num)
            lam += self.lam_step
            valid_num = np.sum(self.v > 1e-4)
            curr_k = len(np.unique(self.labels[self.labels != -1]))
            print(f"迭代{t+1:2d} | λ={lam:.2f} | 有效样本:{valid_num:4d} | 当前簇数:{curr_k}")
        return self.labels

# ====================== 噪声分配辅助函数：噪声点分配到最近簇中心 ======================
def assign_noise_to_nearest_cluster(X_noise, cluster_centers):
    """
    :param X_noise: 噪声样本 (m,d)
    :param cluster_centers: 簇中心 (k,d)
    :return: noise_pred: 噪声点预测标签
    """
    dist = pairwise_distances(X_noise, cluster_centers)
    noise_pred = np.argmin(dist, axis=1)
    return noise_pred

# -------------------------- 主入口【数据集读取完全沿用你原来的写法】 --------------------------
if __name__ == "__main__":
    print("==== MOS噪声过滤 + IHS-DPC聚类 + 噪声回填分配 ====")
    print("=" * 70)

    all_results = {}
    best_params_history = {}

    # MOS参数
    MOS_K = 3
    MOS_ROUND = 3
    NOISE_RATIO = 0.10   # 剔除得分最高20%作为噪声，可修改

    # 数据集列表，和你原来写法保持一致，直接切换
    # dataname=['seeds', 'wine','heart','banknote','landsat','MSRA25', 'Palmdata','usps2d','data_PenDigits2d',"D31", "S1","Aggregation", "R15"]
    # dataname = ['oliver100']
    # dataname = ["Aggregation", "R15"]
    # dataname = ["pathbased", "compound",'jain',"Aggregation", "R15"]
    dataname = ["Aggregation", "R15", 'd31', 's1', "s4", "a3"]
    # dataname = ["unbalance"]

    for name in dataname:
        try:
            # =========【！！！导入方式完全不变，原样保留！！！】=========
            data_path = f'dataset/{name}fed.pkl'
            datapkl = load_dataset(data_path)
            if datapkl is None:
                print(f"跳过数据集 {name}")
                continue

            data = datapkl['full_data']
            true_labels = datapkl['true_label']
            n_all = data.shape[0]
            true_cluster_num = len(np.unique(true_labels))
            params = dataset_params.get(name, {})
            print(f"\n处理数据集: {name} (样本量: {n_all}, 真实类别数: {true_cluster_num})")

            # ========= Step1 MOS计算离群得分 =========
            scaler_global = StandardScaler()
            X_std = scaler_global.fit_transform(data)
            outlier_scores = mos_mean_shift_outlier_score(X_std, k=MOS_K, tune_round=MOS_ROUND)

            # 选出噪声索引：得分最高 NOISE_RATIO 比例
            cut_pos = int(n_all * NOISE_RATIO)
            idx_sorted = np.argsort(outlier_scores)[::-1]
            idx_noise = idx_sorted[:cut_pos]
            idx_clean = idx_sorted[cut_pos:]

            X_clean = data[idx_clean]
            X_noise = data[idx_noise]
            print(f"MOS过滤：干净样本{len(X_clean)}, 噪声样本{len(X_noise)}")

            # ========= Step2 在干净样本上跑 IHS-DPC =========
            model = IterHardScreenDPC(**params)
            pred_clean = model.fit(X_clean, true_cluster_num)

            # 获取IHS-DPC得到的簇中心
            clean_scaler = StandardScaler()
            Xclean_std = clean_scaler.fit_transform(X_clean)
            cluster_centers = Xclean_std[model.center_idx, :]

            # ========= Step3 噪声分配到最近簇 =========
            Xnoise_std = clean_scaler.transform(X_noise)
            pred_noise = assign_noise_to_nearest_cluster(Xnoise_std, cluster_centers)

            # ========= Step4 合并标签，恢复原始顺序 =========
            pred_full = np.zeros(n_all, dtype=int)
            pred_full[idx_clean] = pred_clean
            pred_full[idx_noise] = pred_noise

            # ========= Step5 评估指标 =========
            ari = adjusted_rand_score(true_labels, pred_full)
            nmi = normalized_mutual_info_score(true_labels, pred_full)
            acc = clustering_acc(true_labels, pred_full)

            print(f"{name} 【MOS+IHS-DPC】ARI={ari:.4f}, NMI={nmi:.4f}, ACC={acc:.4f}")

            all_results[name] = {
                "ari": ari,
                "nmi": nmi,
                "acc": acc,
                "pred_labels": pred_full,
                "params": params
            }
            best_params_history[name] = params

        except Exception as e:
            print(f"数据集 {name} 运行异常: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 全部跑完汇总输出
    print("\n" + "="*70)
    print("所有数据集结果汇总【MOS过滤 + IHS-DPC +噪声回填】：")
    for ds, res in all_results.items():
        print(f"{ds:12s} | ARI={res['ari']:.4f} | NMI={res['nmi']:.4f} | ACC={res['acc']:.4f}")
