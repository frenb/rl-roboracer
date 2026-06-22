// TB inside sim-controller runs with
//   --logdir_spec=current:/tmp/active,compare:/tmp/tb_compare
// (see docker-compose.yml + compose/scale.yml). That gives TB TWO
// "experiments":
//   * current/<job_id>/...   <- the live job (trainer writes here)
//   * compare/<job_id>/...   <- on-demand symlinks populated by the
//                              /set_tb_compare_jobs endpoint when
//                              the operator clicks Compare in the
//                              Models tab.
//
// We surface those as TWO TensorBoard tabs that each see only their
// own experiment, by pre-loading a TB regexInput that scopes the
// run-list to one prefix:
//
//   "Tensorboard (current)" - regexInput=^current/   - dedicated to
//      the one currently-training job. Live, never has more than one
//      job because move_all_jobs_data() archives any prior job out
//      of /tmp/active/ at TRAIN pickup.
//
//   "Tensorboard (Analysis)" - regexInput=^compare/  - reserved for
//      cross-job comparisons. Re-pointed on every Models -> Analysis
//      click to ^compare/(<id1>|<id2>|...) so it shows only the
//      selected jobs.
//
// Why TB's regex (vs e.g. two TB processes): one TB process, one
// port, one container layer. Sidebar UX is identical to a "filtered"
// view that the user could also produce by hand. Survives reloads.
const TB_BASE = 'http://localhost:6006';
const TB_CURRENT_FILTER = '^current/';
const TB_COMPARE_FILTER_ALL = '^compare/';
function tbUrl(runFilter) {
  // Pin to the Time Series plugin (#timeseries) instead of the
  // legacy Scalars plugin (#scalars). Time Series is TB's modern
  // unified dashboard and what the operator wants as the default
  // landing - it has the better small-multiple layout for comparing
  // multiple runs and supports the same regex filter we want to use.
  //
  // Param name is `runFilter` (not `regexInput`). TensorBoard
  // renamed it in
  //   https://github.com/tensorflow/tensorboard/pull/5412
  // when adding URL persistence for Time Series; the legacy
  // `regexInput` survives only on the Scalars plugin and is silently
  // ignored by Time Series. Using the wrong name was why the filter
  // appeared to not apply on 2026-05-25 - the URL had
  // ?regexInput=^current/ but the run-list sidebar showed every
  // experiment.
  //
  // The _t cache-bust is what makes setting src to the same logical
  // URL twice (e.g. re-clicking Compare with the same selection)
  // still force an iframe reload. Browsers no-op a same-src write.
  return TB_BASE + '/?runFilter=' + encodeURIComponent(runFilter)
    + '&_t=' + Date.now() + '#timeseries';
}

var config = {
    // Original two-column shape:
    //   * Left column: a single STACK with both Tensorboard tabs
    //     (current + Analysis). The left half of the screen is
    //     dedicated to TB; clicking a tab header switches which
    //     view fills it.
    //   * Right column: sim controller logs (top) + a stack of all
    //     the data / config tabs (Jobs, Models, Leaderboard,
    //     Analysis, Reward Design, Experiment Design).
    //
    // Why a stack on the left (not two separately-mounted iframes
    // stacked vertically): the user wants to see one TB view at a
    // time using the full left half, with tab navigation between
    // them. The Analysis iframe still gets re-pointed by the
    // open-analysis handler whether it's the active tab or not -
    // when the user manually switches to the "Tensorboard (Analysis)"
    // tab after a Compare click, the filtered view is already
    // there.
    content: [
        {
          type: 'row',
          content: [
            {
              type: 'column',
              content: [
                {
                  type: 'stack',
                  // Default-active = current view, since live-
                  // monitoring is the primary use case when the
                  // dashboard opens.
                  activeItemIndex: 0,
                  content: [
                    // Tensorboard (current): pinned to TB's "current"
                    // experiment (= /tmp/active feed). NEVER re-pointed
                    // by the open-analysis handler, so live-training
                    // monitoring survives any Compare click. The
                    // regexInput=^current/ ensures the run sidebar
                    // ONLY ever lists the currently-training job's
                    // runs (eval / train / learner/train / metrics) -
                    // never any archived "compare/..." entries even
                    // if the operator has just clicked Compare.
                    {
                      type: 'component',
                      title: 'Tensorboard (current)',
                      componentName: 'iframeComponent',
                      componentState: {
                        src: tbUrl(TB_CURRENT_FILTER),
                        id: 'tensorboard'
                      }
                    },
                    // Tensorboard (Analysis): id='tensorboard_compare'
                    // (kept for back-compat with the postMessage
                    // routing in goldenlayout's message handler).
                    // The open-analysis handler re-points THIS
                    // iframe's src to '?regexInput=^compare/(id|id|...)
                    // &#scalars' on Compare. Switch to this tab to
                    // see the filtered runs after clicking Analysis
                    // on the Models tab. Default filter is '^compare/'
                    // so before any Compare click it shows the bucket
                    // contents (usually empty after a fresh boot).
                    {
                      type: 'component',
                      title: 'Tensorboard (Analysis)',
                      componentName: 'iframeComponent',
                      componentState: {
                        src: tbUrl(TB_COMPARE_FILTER_ALL),
                        id: 'tensorboard_compare'
                      }
                    },
                    // Mad Scientist Lab: tracks the autonomous
                    // research / proposal / implementation agent
                    // (see rl_agent/madscientist/). Lives in the
                    // same stack as the TB tabs so the operator can
                    // glance at "what is the agent doing right now"
                    // while watching live training metrics in the
                    // adjacent tabs. Phase 0 ships an empty
                    // scaffold; Phase 1 wires the activity feed,
                    // pending-decision cards, and outcomes table.
                    {
                      type: 'component',
                      title: 'Mad Scientist Lab',
                      componentName: 'iframeComponent',
                      componentState: {
                        src: 'http://localhost/madscientist',
                        id: 'madscientist'
                      }
                    }
                  ]
                }
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
                      componentState: { src: 'http://localhost/jobs?v=ws3', id: 'jobs' }
                    },
                    {
                      type: 'component',
                      title: 'Models',
                      componentName: 'iframeComponent',
                      componentState: { src: 'http://localhost/models?v=ws1', id: 'models' }
                    },
                    {
                      type: 'component',
                      title: 'Leaderboard',
                      componentName: 'iframeComponent',
                      componentState: { src: "http://localhost/leaderboard?v=ws3", id: 'leaderboard' }
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
                      title: 'Weakness',
                      componentName: 'iframeComponent',
                      // Per-track-location weakness heatmap (Phase 1
                      // diagnostic). Aggregates logs.position_history
                      // into a 2D spatial grid colored by a blend of
                      // crash rate / slowness / low return. Served by
                      // express.static, so the same ".html" rationale as
                      // Analysis applies: works the moment the file is on
                      // disk, no dashboard image rebuild needed for the
                      // page itself (the /weakness_map data endpoint does
                      // need the server rebuilt).
                      componentState: { src: "http://localhost/weakness.html?v=ws2", id: 'weakness' }
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
                      componentState: { src: "http://localhost/reward_designs.html?v=ws1", id: 'reward_designs' }
                    },
                    {
                      type: 'component',
                      title: 'Experiment Design',
                      componentName: 'iframeComponent',
                      // Schema-driven form for training-loop config:
                      // BC pretrain, replay capacity, optimizer LRs,
                      // network sizes, demo-protected buffer knobs.
                      // Same ".html" rationale as siblings above.
                      componentState: { src: "http://localhost/experiment_designs.html?v=ws1", id: 'experiment_designs' }
                    },
                    {
                      type: 'component',
                      title: 'Gyms',
                      componentName: 'iframeComponent',
                      // Registry of Unity scene / gym configurations.
                      // Gyms appear as a dropdown in the New Job and
                      // Eval Selected dialogs so every job can record
                      // which track geometry it ran against.
                      // ?v= busts the browser cache whenever this number
                      // is incremented after a gyms.html update.
                      componentState: { src: "http://localhost/gyms.html?v=ws1", id: 'gyms' }
                    }
                  ]
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
    // Tell the embedded page to redraw its Tabulator. GoldenLayout sets an
    // inactive tab's iframe to display:none; Tabulator (height:100%) then lays
    // out a zero-height body and DOESN'T recompute when the tab is shown again,
    // so the body sticks all-white. A cross-iframe ResizeObserver can miss this
    // display:none->block transition (Chrome), so we explicitly post a redraw
    // signal on show + resize; the page's TableView listens for it.
    var postRedraw = () => {
      try {
        const ifr = container.getElement().get(0).childNodes[0];
        if (ifr && ifr.contentWindow) {
          ifr.contentWindow.postMessage({ type: 'roboracer:redraw' }, '*');
        }
      } catch (e) { /* ignore */ }
    };
    container.on('resize', () => {
      const iframe = container.getElement().get(0).childNodes[0];
      iframe.width = container.width;
      iframe.height = container.height;
      postRedraw();
    });
    // When this tab becomes active its iframe transitions hidden->shown; the
    // size may not be final on the first 'show' tick, so re-send a couple of
    // times as the layout settles.
    container.on('show', () => {
      postRedraw();
      setTimeout(postRedraw, 60);
      setTimeout(postRedraw, 200);
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
  // SLOT LABELS: when the Models tab sends jobMeta (each entry
  // carries jobId + model.create_date), we sort the entries by
  // create_date ascending and assign labels base / exp1 / exp2 /
  // exp3 / ... - matching exactly what the Analysis tab does in
  // buildBundles -> sortBundlesByCreateDate. The labels then flow
  // through to the symlink basenames the server creates in
  // /tmp/tb_compare/, so TB run names read as:
  //
  //   compare/base_6a09ddd9/{learner,train,eval,metrics}/...
  //   compare/exp1_6a0e9aeb/{learner,train,eval,metrics}/...
  //
  // instead of opaque 24-char ObjectIds. The operator can now map
  // each TB sidebar entry to its Analysis-tab column at a glance.
  //
  // The currently-training job (if a selected model belongs to it)
  // stays under the "current/<full_jobId>/..." prefix because
  // /tmp/active is owned by the trainer and we can't safely rename
  // its dir mid-run. That's an intentional inconsistency - the
  // current/ vs compare/ prefix split itself signals "this one is
  // live".
  try {
    // Target the dedicated 'tensorboard_compare' pane so the training
    // pane (id='tensorboard') stays pinned at the unfiltered view.
    // Fall back to the legacy 'tensorboard' iframe for back-compat if
    // the layout config doesn't define the compare pane (e.g., a user
    // running with an older saved goldenlayout state).
    const tbTarget = iframeRegistry['tensorboard_compare']
      || iframeRegistry['tensorboard'];
    const rawJobIds = Array.isArray(data.jobIds) ? data.jobIds : [];
    const rawJobMeta = Array.isArray(data.jobMeta) ? data.jobMeta : [];

    // Slot label assignment. Sort by create_date ascending so the
    // OLDEST checkpoint is base (this matches the Analysis tab's
    // default ordering, see sortBundlesByCreateDate in
    // analysis.html). Models without a create_date sort to the end
    // (probably broken / partial records).
    //
    // We label up to N=10 slots (base, exp1, exp2, ..., exp9). Beyond
    // 10 we fall back to "exp<N>" but the operator probably has a
    // bigger problem than the sidebar by then.
    const SLOT_LABELS = ['base', 'exp1', 'exp2', 'exp3', 'exp4',
                         'exp5', 'exp6', 'exp7', 'exp8', 'exp9'];
    const sortedMeta = rawJobMeta.slice().sort((a, b) => {
      const ad = a && a.createDate ? Date.parse(a.createDate) : Number.POSITIVE_INFINITY;
      const bd = b && b.createDate ? Date.parse(b.createDate) : Number.POSITIVE_INFINITY;
      if (Number.isNaN(ad) && Number.isNaN(bd)) return 0;
      if (Number.isNaN(ad)) return 1;
      if (Number.isNaN(bd)) return -1;
      return ad - bd;
    });
    const labeledJobs = sortedMeta.map((m, idx) => ({
      jobId: m.jobId,
      label: SLOT_LABELS[idx] || ('exp' + idx),
    }));

    // Body shape: labeled if Models tab supplied jobMeta, legacy
    // {jobIds:[]} otherwise. The server accepts both.
    const setBucketBody = labeledJobs.length
      ? { jobs: labeledJobs }
      : { jobIds: rawJobIds };

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
      body: JSON.stringify(setBucketBody),
    }).then((r) => {
      if (!r.ok) console.warn('set_tb_compare_jobs HTTP ' + r.status);
      return r.ok ? r.json() : null;
    }).catch((e) => {
      console.warn('set_tb_compare_jobs failed:', e);
      return null;
    });

    setBucket.then((bucketResp) => {
      if (!tbTarget || !tbTarget.iframe) return;

      // Use the SERVER's view of which job_ids actually got linked,
      // INCLUDING the symlink basenames it assigned (labeled or not).
      // bucketResp.linkNames is { jobId -> "<label>_<short>" | "<full>" }
      // for every jobId successfully symlinked.
      //
      // We anchor the regex to ^(current|compare)/ so the Analysis
      // tab shows ONLY the user's selected jobs (no leakage from
      // other archived runs in the bucket). The currently-training
      // job (if a selected model belongs to it) shows up under
      // current/<full_jobId>/...; archived jobs show up under
      // compare/<label>_<short>/...
      const linkNames = (bucketResp && bucketResp.linkNames) || {};
      // Build the alternation: prefer the labeled symlink basename
      // (for archived jobs) and fall back to the full jobId (for
      // currently-training jobs not in /tmp/jobsdata yet).
      const alternationSegments = rawJobIds.map((id) => {
        return linkNames[String(id)] || String(id);
      });
      // Also include any current/<full_jobId>/ matches for jobs
      // whose live data is in /tmp/active (they won't have a
      // labeled symlink but they should still show up if selected).
      const currentSegments = rawJobIds.filter((id) =>
        !linkNames[String(id)] // not in compare bucket = probably live
      );

      // Escape regex metacharacters in the labels / ids before joining.
      const escapeRe = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const compareAlt = alternationSegments
        .map(escapeRe)
        .filter((s) => s.length);
      const currentAlt = currentSegments
        .map(String)
        .map(escapeRe)
        .filter((s) => s.length);

      let nextSrc;
      if (compareAlt.length || currentAlt.length) {
        // We build a single alternation with two anchored subgroups:
        //   ^compare/(label_short|...|label_short)   - matches archived
        //   ^current/(full_jobId|...)                - matches live
        // Joined into one regex so a mixed selection (some live, some
        // archived) renders cleanly.
        const parts = [];
        if (compareAlt.length) {
          parts.push('^compare/(' + compareAlt.join('|') + ')');
        }
        if (currentAlt.length) {
          parts.push('^current/(' + currentAlt.join('|') + ')');
        }
        const pattern = parts.length > 1
          ? '(' + parts.join('|') + ')'
          : parts[0];
        nextSrc = tbUrl(pattern);
      } else {
        // No filterable job_ids in this selection (every model was
        // legacy/pre-feature). Drop down to the default "^compare/"
        // filter so the user at least sees the bucket contents
        // (likely empty for this selection) rather than the full
        // current+compare experiment soup.
        nextSrc = tbUrl(TB_COMPARE_FILTER_ALL);
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
