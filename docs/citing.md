# Citing jbubble

If jbubble contributes to work that you publish, cite the software. The
repository's [`CITATION.cff`](https://github.com/imperial-nsb/jbubble/blob/main/CITATION.cff)
file holds the citation metadata, including the authors and the version.

To get a formatted citation, open the
[repository on GitHub](https://github.com/imperial-nsb/jbubble) and select
**Cite this repository** in the sidebar. GitHub formats the metadata in
`CITATION.cff` as APA or BibTeX. Reference managers such as Zotero also read
the file.

Cite the version that you used, so that others can reproduce your results.
To print the installed version, run the following command:

```bash
python -c "import jbubble; print(jbubble.__version__)"
```

jbubble was first presented at the 2026 IEEE International Ultrasonics
Symposium (IUS). There's no jbubble paper to cite yet, so cite the software
itself.

The physical models in jbubble come from the literature. When your results
depend on a model, also cite the paper behind it; each class in the
[API reference](api/index.md) lists its references.
