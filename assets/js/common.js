$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  $("a").removeClass("waves-effect waves-light");

  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });
    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    Toc.init({
      $nav: $myNav,
      $scope: $("#markdown-content"),
    });
    if (!$myNav.find(".nav-link").length) {
      $(".toc-sidebar").hide();
    }
    $("body").scrollspy({
      target: navSelector,
      offset: 80,
    });
    $("body").scrollspy("refresh");
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let theme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (theme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // Collapsible boxes functionality
  const initCollapsibles = () => {
    document.querySelectorAll('.collapsible-header').forEach(header => {
      header.addEventListener('click', toggleCollapsible);
      header.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleCollapsible.call(header);
        }
      });
    });
  };

  function toggleCollapsible() {
    const content = this.nextElementSibling;
    const isOpen = this.classList.contains('open');
    
    this.classList.toggle('open');
    content.classList.toggle('open');
    this.setAttribute('aria-expanded', !isOpen);
    
    // Update icon rotation instead of changing the icon class
    const icon = this.querySelector('.collapsible-icon');
    if (icon) {
      // The CSS handles the rotation based on the .open class
    }
  }

  initCollapsibles();
});
