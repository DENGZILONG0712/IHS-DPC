# 数据标准化。标准版。



import numpy as np
import pickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# -------------------------- 数据集工具函数 --------------------------
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

# 各数据集完整超参配置
# 固定开启硬筛选，移除hard_sp参数；max_iter、lam_step全局固定
dataset_params = {
    # 小标准数据集
    "iris": {"dc_ratio": 0.03, "init_lam": 0.15},
    "ecoli": {"dc_ratio": 0.028, "init_lam": 0.16},
    "thyroid": {"dc_ratio": 0.022, "init_lam": 0.17},
    "seeds": {"dc_ratio": 0.032, "init_lam": 0.14},
    "zoo": {"dc_ratio": 0.04, "init_lam": 0.12},
    "wine": {"dc_ratio": 0.026, "init_lam": 0.19},
    # 图像/高维2D降维数据集
    "mnist2d": {"dc_ratio": 0.02, "init_lam": 0.2},
    "usps2d": {"dc_ratio": 0.025, "init_lam": 0.18},
    "data_PenDigits2d": {"dc_ratio": 0.023, "init_lam": 0.19},
    "MSRA25": {"dc_ratio": 0.021, "init_lam": 0.21},
    "Palmdata": {"dc_ratio": 0.024, "init_lam": 0.20},
    # 文本/分类大数据集
    "spambase": {"dc_ratio": 0.018, "init_lam": 0.22},
    "msplice": {"dc_ratio": 0.020, "init_lam": 0.21},
'oliver100': {"dc_ratio": 0.9999, "init_lam": 8}


}

# -------------------------- 迭代硬筛选密度峰值聚类 IHS-DPC --------------------------
class IterHardScreenDPC:
    def __init__(self, dc_ratio=0.02, init_lam=0.1):
        self.dc_ratio = dc_ratio
        self.init_lam = init_lam
        # 全局固定参数
        self.lam_step = 0.1
        self.max_iter = 20

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
        loss = rho  # 去除负号，loss直接等于密度ρ
        n = len(rho)
        v = np.zeros(n)
        # 大于阈值保留，小于等于剔除
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
        # Z-score标准化
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

# -------------------------- 主入口 --------------------------
if __name__ == "__main__":
    print("开始精简版聚类测试...")
    print("=" * 60)

    all_results = {}
    best_params_history = {}

    dataname=['seeds', 'wine','heart','banknote','landsat','MSRA25', 'Palmdata','usps2d','data_PenDigits2d']
    dataname = ['oliver100']
    for name in dataname:
        try:
            data_path = f'dataset/{name}fed.pkl'
            datapkl = load_dataset(data_path)
            if datapkl is None:
                print(f"跳过数据集 {name}")
                continue

            data = datapkl['full_data']
            true_labels = datapkl['true_label']

            print(f"\n处理数据集: {name} (样本量: {data.shape[0]}, 真实类别数: {len(np.unique(true_labels))})")
            true_cluster_num = len(np.unique(true_labels))
            params = dataset_params.get(name, {})

            model = IterHardScreenDPC(**params)
            pred_labels = model.fit(data, true_cluster_num)
            print(pred_labels)

            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            print(f"{name} 聚类结果 ARI={ari:.4f}, NMI={nmi:.4f}")

            all_results[name] = {
                "ari": ari,
                "nmi": nmi,
                "pred_labels": pred_labels,
                "params": params
            }
            best_params_history[name] = params

        except Exception as e:
            print(f"数据集 {name} 运行异常: {e}")
            continue

    # 全部跑完汇总输出
    print("\n" + "="*60)
    print("所有数据集结果汇总：")
    for ds, res in all_results.items():
        print(f"{ds:8s} | ARI={res['ari']:.4f} | NMI={res['nmi']:.4f}")