// Render the math that pymdownx.arithmatex (generic mode) wraps in \( \) and
// \[ \]. With instant navigation, document$ emits on every page load, so math
// renders after each navigation too.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
