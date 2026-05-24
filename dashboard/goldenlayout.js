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
                      // id: 'tensorboard' registers this iframe in
                      // iframeRegistry so the open-analysis message
                      // handler below can re-point it at a regex-
                      // filtered URL when the user clicks "Compare in
                      // Analysis" on N selected models. Without an id
                      // the handler can't find the iframe.
                      componentState: { src: 'http://localhost:6006', id: 'tensorboard' }
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
                    },
                    {
                      type: 'component',
                      title: 'Reward Design',
                      componentName: 'iframeComponent',
                      // Same ".html" rationale as Analysis: served by
                      // express.static the moment the file lands on
                      // disk, so the iframe works without waiting for
                      // a dashboard image rebuild that registers the
                      // matching /reward_designs route.
                      componentState: { src: "http://localhost/reward_designs.html", id: 'reward_designs' }
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
// ---------------------------------------------------------------- *
// Cross-iframe routing: Models tab "Job ID" cross-link
//
// When the user clicks a Job ID cell on the Models tab, the page
// posts {type: 'roboracer:open-job', jobId: '<id>'}. We:
//   1. Activate the Jobs tab in its GoldenLayout stack.
//   2. Forward the message into the Jobs iframe so the Jobs page can
//      scroll its Tabulator viewport to the matching row and flash
//      it (see jobs.html's window.message listener).
//
// Robustness: if the Jobs iframe hasn't finished loading when the
// message arrives, the page's first-load handler also reads from
// sessionStorage (same pattern as the open-analysis handoff).
// ---------------------------------------------------------------- *
window.addEventListener('message', function (ev) {
  const data = ev && ev.data;
  if (!data || typeof data !== 'object') return;
  if (data.type === 'roboracer:open-job') {
    const target = iframeRegistry['jobs'];
    if (!target) {
      console.warn('Jobs tab not registered yet; ignoring open-job');
      return;
    }
    // Activate the Jobs tab in its parent stack.
    try {
      const item = target.container.parent;
      if (item && item.parent && typeof item.parent.setActiveContentItem === 'function') {
        item.parent.setActiveContentItem(item);
      }
    } catch (err) {
      console.error('Failed to activate Jobs tab:', err);
    }
    // Persist for first-load lookup. We use a key distinct from the
    // analysis handoff so the two flows never step on each other.
    try {
      sessionStorage.setItem(
        'roboracer:jobs-focus',
        JSON.stringify({ jobId: String(data.jobId || ''), ts: Date.now() }));
    } catch (_e) { /* swallow */ }
    // Direct postMessage into the Jobs iframe. Same boot-race
    // double-fire pattern as the analysis handoff: send now AND on
    // next 'load' to cover the case where the iframe is still
    // hydrating.
    try {
      if (target.iframe && target.iframe.contentWindow) {
        target.iframe.contentWindow.postMessage(data, '*');
      }
    } catch (err) {
      console.error('Failed to forward open-job to iframe:', err);
    }
    try {
      const ifr = target.iframe;
      if (ifr && !ifr.__roboracerJobDeferred) {
        ifr.__roboracerJobDeferred = true;
        ifr.addEventListener('load', function () {
          try {
            if (ifr.contentWindow) ifr.contentWindow.postMessage(data, '*');
          } catch (_e) { /* swallow */ }
        }, { once: true });
      }
    } catch (_err) {
      /* swallow */
    }
    return;
  }

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

  // ---- TensorBoard handoff -------------------------------------- *
  //
  // Two-stage: first we ask the dashboard server to symlink the
  // selected jobs' archived run directories into TB's on-demand
  // compare bucket (/tmp/tb_compare); then we re-point the TB iframe
  // at a regex-filtered URL so its scalars view zooms straight to
  // those runs.
  //
  // Why the bucket: TB only renders runs in the directories it's
  // scanning. The sim-controller container's TB process scans
  // /tmp/active (the currently-training job) and /tmp/tb_compare
  // (initially empty, this endpoint populates it on demand). It does
  // NOT scan /tmp/jobsdata (where the trainer archives finished
  // jobs), so by default the unfiltered TB sidebar stays minimal -
  // just the live job. The compare bucket gives the user temporary
  // visibility into specific historical jobs without permanently
  // bloating the sidebar.
  //
  // Run name layout after the symlink:
  //   current/<live_job_id>/{learner,train,eval,metrics}/...
  //   compare/<archived_job_id>/{learner,train,eval,metrics}/...
  //
  // The regex filter matches "<job_id>" anywhere in the run name so
  // it picks up both the "current/" prefix (if the user is comparing
  // a model from the currently-training job) and the "compare/"
  // prefix.
  //
  // jobIds is provided by the Models tab. Legacy models without a
  // job_id are filtered out there, so jobIds[] is always populated
  // with real job ids (potentially fewer than modelIds.length - see
  // the warning toast on the Models tab).
  try {
    const tbTarget = iframeRegistry['tensorboard'];
    const rawJobIds = Array.isArray(data.jobIds) ? data.jobIds : [];

    // Fire-and-await the server-side symlink shuffle. We deliberately
    // chain the iframe reload on its completion so TB doesn't reload
    // a half-second before the symlinks land (TB's --reload_interval
    // of 5s would otherwise leave the user staring at "no scalars"
    // for several seconds). If the POST fails, fall back to setting
    // the iframe URL anyway so the user at least sees the filtered
    // view of whatever's already in the bucket.
    const setBucket = fetch('/set_tb_compare_jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobIds: rawJobIds }),
    }).then((r) => {
      if (!r.ok) console.warn('set_tb_compare_jobs HTTP ' + r.status);
      return r.ok ? r.json() : null;
    }).catch((e) => {
      console.warn('set_tb_compare_jobs failed:', e);
      return null;
    });

    setBucket.then((bucketResp) => {
      if (!tbTarget || !tbTarget.iframe) return;

      // Use the SERVER's view of which job_ids actually got linked.
      // If a job's archived directory wasn't on disk (e.g., the
      // currently-training job whose data lives in /tmp/active, or
      // a model whose archive was manually deleted), the server's
      // `linked` list will be a subset of what the dashboard sent.
      // We still want to include the originally-requested ids in
      // the regex so currently-active jobs show up under the
      // "current/" prefix - they just don't need a symlink.
      const escaped = rawJobIds
        .map(String)
        .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .filter((s) => s.length);

      let nextSrc;
      if (escaped.length) {
        const pattern = escaped.join('|');
        // Cache-bust on _t so re-clicking with the same selection
        // still forces an iframe reload (otherwise setting src to
        // an identical URL is a no-op in most browsers).
        nextSrc =
          'http://localhost:6006/?regexInput=' +
          encodeURIComponent(pattern) +
          '&_t=' + Date.now() +
          '#scalars';
      } else {
        // No filterable job_ids in this selection (every model was
        // legacy/pre-feature). Drop the filter so the user at least
        // sees the unfiltered view rather than a broken regex.
        nextSrc = 'http://localhost:6006/?_t=' + Date.now() + '#scalars';
      }
      tbTarget.iframe.src = nextSrc;

      // If the server reported missing job archives, log a
      // breadcrumb so we can debug "I selected 3 models but only
      // see 2 in TB" later. Most common reason: one of the models
      // is from the currently-training job (no archive yet).
      if (bucketResp && Array.isArray(bucketResp.missing) && bucketResp.missing.length) {
        console.info(
          'TensorBoard compare: ' + bucketResp.missing.length +
          ' selected job(s) had no archived TB data (likely currently-' +
          'training jobs - they show up under "current/" instead): ' +
          bucketResp.missing.join(', '));
      }
    });
  } catch (err) {
    console.error('Failed to re-point Tensorboard iframe:', err);
  }
});
