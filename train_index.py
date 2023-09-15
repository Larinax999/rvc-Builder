import numpy as np,os,faiss
from multiprocessing import cpu_count
from sklearn.cluster import MiniBatchKMeans
from Settings import exp_dir,name

feature_dir = f"{exp_dir}/3_feature"
npys = []

def main():
    for v in sorted(list(os.listdir(feature_dir))):npys.append(np.load(f"{feature_dir}/{v}"))
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        big_npy = MiniBatchKMeans(n_clusters=10000,verbose=True,batch_size=256*cpu_count,compute_labels=False,init="random").fit(big_npy).cluster_centers_
    np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(768,f"IVF{n_ivf},Flat") # "IVF%s,PQ128x4fs,RFlat"%n_ivf
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    print("training...")
    index.train(big_npy)
    faiss.write_index(index,f"./index/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{name}_v2.index") # ./index/{name}/
    print("write trained.index")

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(index,f"./index/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{name}_v2.index")
    print("write added.index")

if __name__ == "__main__":
    # os.makedirs(f"./index/{name}",exist_ok=True)
    main()
    print("All Done")