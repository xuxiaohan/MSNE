# MSNE

## A Network Embedding Based Method for Partial Multi-Omics Integration in Cancer Subtyping.

Integrative analysis of multiple omics offers the opportunity to uncover coordinated cellular processes acting across different omics layers. The ever-increasing of multi-omics data provides us a comprehensive insight into cancer subtyping. Many multi-omics integrative methods have been developed, but few of them can deal with partial datasets in which some samples have data for a subset of the omics. In this study, we propose a partial multi-omics integrative method, MSNE (Multiple Similarity Network Embedding), for cancer subtyping. MSNE integrates the multi-omics information by embedding the neighbor relations of samples defined by the random walk on multiple similarity networks. We compared MSNE with five existing multi-omics integrative methods on twelve datasets in both full and partial scenarios. MSNE achieved the best result on pan-cancer and image datasets. Furthermore, on ten cancer subtyping datasets, MSNE got the most enriched clinical parameters and comparable log-rank test P-values in survival analysis. In conclusion, MSNE is an effective and efficient integrative method for multi-omics data and, especially, has a strong power on partial datasets.

## Highlights

* Imputation or filtration on partial datasets leads to worse integration performance.

* We propose MSNE, a network embedding based integration method of partial omics data.

* MSNE can capture the similarity of samples that do not appear in any common omics.

* MSNE outperforms other integration methods on both full and partial datasets.

* MSNE can be used as a feature extraction method for other downstream analysis.


## Version

1.0.0

## Author

Han Xu, Lin Gao, Mingfeng Huang, Ran Duan.

## Maintainer

Han Xu <myxuxiaohan@outlook.com>

## How to use
```python
MSNE(views, n_clusters=5, k=20,workers=4, walk_length=20, num_walks=100, embed_size=100, window_size=10)
```
MSNE is a multi-omics integrative clustering method for cancer subtyping, especially when the
    multi-omics dataset is partial (e.g. some samples have only a subset of omics data). MSNE construct
    similarity network for each omics data, and then embedding the multiple similarity networks to
    d-dimensional vector space. Kmeans is used to cluster the samples finally.

    :param views: the list of pandas.DataFrame(i.e. omics data). each row in omics data is a sample, each column in omics
     data is a feature. the index of omics data will be considered as the name of sample.

    :param n_clusters: int, default 10. The number of clusters for Kmeans.

    :param k: int, default 20. The top k neighborhoods of each node will be treated as local neighbors.

    :param workers: int, default 4. The number of parallel threads.

    :param walk_length: int,default 20. The length of sequences generated by random walk on multiple networks.

    :param num_walks: int, default 100. Starting with each node, MSNE will generate 'num_walks' sequences.

    :param embed_size: int, default 100. the dimension of embedding vectors.

    :param window_size: int, default 10. the window_size in skip-gram.

    :return: The dict with elements:
        embeddings: pandas.DataFrame, the low dimensional vector representation of each samples.
        group: pandas.DataFrame, the clustering of samples.

###example:
```python
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
```


