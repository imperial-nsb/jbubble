// Render the math that pymdownx.arithmatex (generic mode) writes as \( \) and
// \[ \]. It converts $...$ and $$...$$ before this runs, so a bare dollar sign
// in prose stays text. With instant navigation, document$ emits on every page
// load, so math renders after each navigation too.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
