var config = {
    content: [
        {
          type: 'row',
          content: [
            {
                type: 'column',
                content: [
                   
                    {
                      type: 'component',
                      title: 'Tensorboard',
                      componentName: 'iframeComponent',
                      componentState: { src: 'http://localhost:6006' }
                    },
                ]
            },
            {
              type: 'column',
              content: [
                {
                  type: 'component',
                  title: 'sim controller logs',
                  componentName: 'iframeComponent',
                  componentState: { src: 'http://localhost/logs' }
                },
                {
                  type: 'stack',
                  content: [
                    {
                      type: 'component',
                      title: 'Jobs',
                      componentName: 'iframeComponent',
                      componentState: { src: 'http://localhost/jobs', id: 'jobs' }
                    },
                    {
                      type: 'component',
                      title: 'Models',
                      componentName: 'iframeComponent',
                      componentState: { src: 'http://localhost/models', id: 'models' }
                    },
                    {
                      type: 'component',
                      title: 'Leaderboard',
                      componentName: 'iframeComponent',
                      componentState: { src: "http://localhost/leaderboard", id: 'leaderboard' }
                    },
                    {
                      type: 'component',
                      title: 'Analysis',
                      componentName: 'iframeComponent',
                      // The Analysis tab boots empty; the user gets it
                      // populated by selecting 2+ rows in the Models tab
                      // and clicking "Compare in Analysis". That click
                      // posts a message which is routed below to switch
                      // tabs and feed the selection into this iframe.
                      //
                      // We deliberately request the static file directly
                      // (".html") rather than the prettier "/analysis"
                      // route. The other tabs use clean URLs because the
                      // dashboard server registers explicit routes for
                      // them at build time. The Analysis route is newer
                      // and may not exist on a running container until
                      // the dashboard image is rebuilt - but the HTML
                      // file is bind-mounted and is served by the
                      // express.static middleware as-is, so requesting
                      // ".html" lets the tab work the moment the static
                      // files appear on disk, without needing the server
                      // process to be restarted.
                      componentState: { src: "http://localhost/analysis.html", id: 'analysis' }
                    }]
                }
              ]
              }
          ]
        }
    ]
};
var myLayout = new GoldenLayout( config );

// Registry of iframe-based components keyed by their componentState.id.
// Populated as GoldenLayout instantiates each iframeComponent below.
// Used to (a) find an iframe's container so we can activate its stack
// and (b) postMessage into the iframe's contentWindow when a sibling
// tab wants to hand off state (e.g. Models -> Analysis selection).
var iframeRegistry = {};

var iframeComponent = function(container, componentState) {
    container.on('resize', () => {
      const iframe = container.getElement().get(0).childNodes[0];
      iframe.width = container.width;
      iframe.height = container.height;
    });
    // This code seems to run only once; attach .on event handlers to react
    // to changes, don't expect this code to be rerun.
    console.log("componentState.src: " + componentState.src);
    const newChild = document.createElement("iframe")
    newChild.frameBorder=0;
    // Iframe outer background is what bleeds through during the inner
    // page's load (most visible for the TensorBoard iframe, which can
    // take a couple of seconds to hydrate) and through any scrollbar
    // / layout gaps in the inner page. Match the slate-950-ish dark
    // background used as the page bg in jobs/models/leaderboard/logs
    // so the iframe never flashes white on load and reads as part of
    // the same dark surface as the GoldenLayout shell. TensorBoard's
    // own dark mode then paints its UI on top of this base.
    newChild.style = "background:#0b1120;"
    newChild.src=componentState.src;
    container
      .getElement()
      .get(0)
      .appendChild(newChild);

    // Stash a handle to this iframe so cross-tab handoffs (see
    // window.message listener below) can find it by id. We register
    // both the container (for activating the stack) and the iframe
    // element (for postMessage routing into the page itself).
    if (componentState.id) {
      iframeRegistry[componentState.id] = {
        container: container,
        iframe: newChild,
      };
    }
}

var simpleComponent = function(container, componentState) {
    const newChild = document.createElement("h2");
    newChild.innerText = componentState.label;
    container
      .getElement()
      .get(0)
      .appendChild(newChild);
}

myLayout.registerComponent('iframeComponent', iframeComponent);
myLayout.registerComponent('simpleComponent', simpleComponent);

myLayout.init();

// ---------------------------------------------------------------- *
// Cross-iframe routing
//
// The Models tab posts {type: 'roboracer:open-analysis', modelIds}
// to its parent (this window) when the user clicks "Compare in
// Analysis". We:
//
//   1. Activate the Analysis tab in its GoldenLayout stack so it
//      becomes visible.
//   2. Forward the same message into the Analysis iframe's
//      contentWindow so the analysis page can render the selection.
//
// The Analysis page also caches the last selection in
// sessionStorage; we forward immediately AND let the page re-pull
// from sessionStorage when it (re)loads. This makes the handoff
// robust to whichever order things finish booting: the iframe may
// still be loading its bundle when the message arrives.
// ---------------------------------------------------------------- *
window.addEventListener('message', function (ev) {
  const data = ev && ev.data;
  if (!data || typeof data !== 'object') return;
  if (data.type !== 'roboracer:open-analysis') return;

  const target = iframeRegistry['analysis'];
  if (!target) {
    console.warn('Analysis tab not registered yet; ignoring open-analysis');
    return;
  }

  // Activate the Analysis tab in its parent stack so the user sees
  // it switch in front of them. GoldenLayout puts the immediate
  // parent of a component-content-item at .parent; for tabs in a
  // stack that's the stack itself.
  try {
    const item = target.container.parent;
    if (item && item.parent && typeof item.parent.setActiveContentItem === 'function') {
      item.parent.setActiveContentItem(item);
    }
  } catch (err) {
    console.error('Failed to activate Analysis tab:', err);
  }

  // Persist the selection so the analysis page can pick it up on its
  // own initial load (handles the case where the iframe hasn't yet
  // installed its message listener when the click happened).
  try {
    sessionStorage.setItem(
      'roboracer:analysis-selection',
      JSON.stringify({
        modelIds: Array.isArray(data.modelIds) ? data.modelIds : [],
        ts: Date.now(),
      }));
  } catch (err) {
    // sessionStorage may be unavailable in odd contexts (e.g. private
    // mode quirks); the postMessage path below is the primary handoff
    // so we can safely swallow this.
    console.warn('sessionStorage write failed:', err);
  }

  // Direct postMessage into the analysis iframe (its window listener
  // re-renders on receipt). Same-origin so no targetOrigin gymnastics
  // - both iframes are served from this dashboard server.
  //
  // We may end up sending to an iframe that hasn't finished loading
  // its analysis.html yet (about:blank), in which case the message is
  // dropped on the floor by the receiver. To make the handoff robust
  // we also fire the message again once on the next 'load' event of
  // the iframe. The iframe page will dedupe-by-id internally (and
  // also reads sessionStorage as a third independent path).
  try {
    if (target.iframe && target.iframe.contentWindow) {
      target.iframe.contentWindow.postMessage(data, '*');
    }
  } catch (err) {
    console.error('Failed to forward open-analysis to iframe:', err);
  }
  try {
    const ifr = target.iframe;
    if (ifr && !ifr.__roboracerDeferred) {
      ifr.__roboracerDeferred = true;
      ifr.addEventListener('load', function () {
        try {
          if (ifr.contentWindow) ifr.contentWindow.postMessage(data, '*');
        } catch (e) { /* swallow */ }
      }, { once: true });
    }
  } catch (err) {
    /* swallow - the sessionStorage fallback still catches us */
  }
});
