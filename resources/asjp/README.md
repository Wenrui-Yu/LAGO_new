# ASJP LDND Source

`output.txt` is the ASJP62 LDND output used to derive the AJSP topology.

The parent source files are:

- `../ASJPSoftware003/data.txt`: ASJP word-list input.
- `../ASJPSoftware003/asjp62.f`: Fortran implementation.
- `../ASJPSoftware003/output.txt`: LDND lower-triangular distance matrix.

The ASJP62 calculation uses normalized Levenshtein distance between ASJP
transcriptions for shared concepts. For each language pair, the program
accumulates:

- `PMJ`: summed normalized Levenshtein distance over same-concept word pairs.
- `PNJ`: number of same-concept comparisons.
- `QMJ`: summed normalized Levenshtein distance over different-concept word pairs.
- `QNJ`: number of different-concept comparisons.

The printed LDND value is:

```text
100 * PMJ * QNJ / (PNJ * QMJ)
```

This is why closely related pairs such as English-Dutch and German-Dutch have
lower values than unrelated pairs.
