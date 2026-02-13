window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  chtml: {
    matchFontHeight: false,
    mtextInheritFont: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
    renderActions: {
      addMenu: []
    }
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
