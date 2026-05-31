# Lang2vec Syntactic Distance Source

`syntactic_distances.csv` is a fallback cache of the lang2vec syntactic
distance output for the 14-language set used in the original topology scripts.

The primary source is the local parent directory:

- `../lang2vec-master/lang2vec/lang2vec.py`
- `../lang2vec-master/lang2vec/data/distances2.zip`
- inside the zip: `syntactic_upper_round2_sparse.npz`
- language order: `eng,zho,fra,deu,ind,ita,por,rus,spa,ara,nld,hin,jpn,vie`

The values are obtained with:

```python
import lang2vec.lang2vec as l2v

langs = ["eng", "zho", "fra", "deu", "ind", "ita", "por", "rus",
         "spa", "ara", "nld", "hin", "jpn", "vie"]
matrix = l2v.distance("syntactic", langs)
```

The lang2vec README describes these distances as precomputed typological
distances; in most cases they correspond to cosine distances between the
corresponding feature vectors. The syntactic graph used here thresholds this
distance matrix at `0.45`.
