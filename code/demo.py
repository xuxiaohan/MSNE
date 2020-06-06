import pandas as pd
from embedding import MSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi

if __name__ == '__main__':

    # the ground truth
    samples=[f"sample_{i}" for i in range(2000)]
    label=pd.Series([i for i in range(10) for j in range(200)], index=samples)

    # read the Pixel and Fourier views of handwritten datasets.
    view1=pd.read_csv("../data/handwritten/mfeat-fou.csv", index_col=0)
    view2=pd.read_csv("../data/handwritten/mfeat-pix.csv", index_col=0)

    #apply MSNE on the multi-view dataset.
    result=MSNE([view1,view2],
                n_clusters=10, k=20, workers=4,
                walk_length=20, num_walks=20,
                embed_size=100, window_size=10)

    #sort the samples by name
    embeddings=result["embeddings"].reindex(samples)
    group=result["group"].reindex(samples).values.reshape(-1)

    #show the result of MSNE
    low=TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(low[:,0],low[:,1], c=group)
    print("NMI is:", nmi(label,group, average_method='geometric'))
