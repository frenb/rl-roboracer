/* ---------------------------------------------------------------- *
 * rl-roboracer dashboard — shared components & utilities
 *
 * Vanilla ES2020. No bundler, no framework. All rendered HTML uses
 * Tailwind utility classes (Tailwind Play CDN is loaded by each
 * tab's <head> before this script).
 *
 * Public surface:
 *
 *   formatters.statusBadge(status)        -> badge HTML for NOT_STARTED/IN_PROGRESS/DONE
 *   formatters.jobTypeBadge(type)         -> badge HTML for TRAIN/DEMO/EVAL/BC_...
 *   formatters.modelTypeBadge(type)       -> badge HTML for SacAgent/GreedyPolicy/RandomPyPolicy
 *   formatters.robotTypeBadge(type)       -> badge HTML for robotaxi/niryo
 *   formatters.percentBar(fraction0to1)   -> progress bar HTML
 *   formatters.relativeTime(date)         -> "5m ago", "2 days ago"
 *   formatters.shortId(objectId)          -> last 8 chars of ObjectId, mono-spaced
 *   formatters.truncatePath(s)            -> path with mono+ellipsis cell
 *   formatters.number(v, opts)            -> tabular-nums-formatted number cell
 *
 *   showToast({type, message, duration})  -> renders a transient toast
 *   confirmDialog({title, message, ...})  -> Promise<boolean>
 *
 *   class TableView                       -> declarative table component
 *
 * TableView usage example:
 *
 *   const tv = new TableView(rootEl, {
 *     columns: [
 *       { key: '_id',     label: 'ID',     sortable: true,  filter: true,
 *         render: (v) => formatters.shortId(v) },
 *       { key: 'status',  label: 'Status', sortable: true,  filter: true,
 *         render: (v) => formatters.statusBadge(v) },
 *       ...
 *     ],
 *     selectable: true,
 *     rowKey: (row) => row._id,
 *     pageSize: 25,
 *     emptyMessage: 'No jobs yet.',
 *   });
 *   tv.onSelectionChange((selectedRows) => updateButtons(selectedRows));
 *   tv.setData(jsonRows);
 * ---------------------------------------------------------------- */

(function (global) {
  "use strict";

  // Version banner. Logged once per iframe so the user can confirm in
  // DevTools console which build of components.js is actually running.
  // Bump alongside the ?v=... query string in jobs/models/leaderboard/
  // analysis HTML when shipping a change to this file.
  console.log('[RoboracerUI] components.js v20260516-selectcol loaded');

  /* ---------- formatters ---------------------------------------- */

  const BADGE_BASE =
    'inline-flex items-center px-2 py-0.5 rounded-full ' +
    'text-[10px] font-semibold uppercase tracking-[0.04em] whitespace-nowrap leading-5';

  const BADGE_TONE = {
    success: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    warning: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    info:    'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    neutral: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
    danger:  'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    indigo:  'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300',
  };

  function badge(tone, text) {
    return `<span class="${BADGE_BASE} ${BADGE_TONE[tone] || BADGE_TONE.neutral}">${escape(text)}</span>`;
  }

  const formatters = {
    statusBadge(status) {
      const tone = ({
        NOT_STARTED: 'neutral',
        IN_PROGRESS: 'info',
        DONE: 'success',
      })[status] || 'neutral';
      return badge(tone, status || '?');
    },

    jobTypeBadge(type) {
      const tone = ({
        TRAIN: 'indigo',
        DEMO: 'warning',
        EVAL: 'neutral',
        BC_TRAINING_ONLY: 'warning',
      })[type] || 'neutral';
      return badge(tone, type || '?');
    },

    modelTypeBadge(type) {
      const tone = ({
        SacAgent: 'indigo',
        GreedyPolicy: 'neutral',
        RandomPyPolicy: 'neutral',
      })[type] || 'neutral';
      return badge(tone, type || '?');
    },

    robotTypeBadge(type) {
      return badge('neutral', type || '?');
    },

    percentBar(fraction) {
      const f = Math.max(0, Math.min(1, Number(fraction) || 0));
      const pct = (f * 100).toFixed(0);
      const complete = f >= 0.9999;
      const fillClasses = complete
        ? 'bg-gradient-to-b from-emerald-400 to-emerald-600 rounded-full'
        : 'bg-gradient-to-b from-indigo-400 to-indigo-600 rounded-l-full';
      return `
        <div class="relative w-full min-w-[100px] max-w-[220px] h-4 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden shadow-inner">
          <div class="h-full ${fillClasses} transition-[width] duration-200" style="width: ${pct}%"></div>
          <div class="absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-slate-900 dark:text-slate-100 mix-blend-luminosity">${pct}%</div>
        </div>`;
    },

    relativeTime(value) {
      if (!value) return '<span class="text-slate-500 dark:text-slate-400 whitespace-nowrap">—</span>';
      const d = (value instanceof Date) ? value : new Date(value);
      if (isNaN(d.getTime())) return '<span class="text-slate-500 dark:text-slate-400 whitespace-nowrap">—</span>';
      const seconds = Math.round((Date.now() - d.getTime()) / 1000);
      const abs = Math.abs(seconds);
      let label;
      if (abs < 45)         label = `${seconds}s ago`;
      else if (abs < 60*45) label = `${Math.round(seconds / 60)}m ago`;
      else if (abs < 3600*22) label = `${Math.round(seconds / 3600)}h ago`;
      else if (abs < 86400*30) label = `${Math.round(seconds / 86400)}d ago`;
      else label = d.toLocaleDateString();
      const iso = escape(d.toISOString());
      return `<span class="text-slate-600 dark:text-slate-300 whitespace-nowrap" title="${iso}">${escape(label)}</span>`;
    },

    shortId(id) {
      // `whitespace-nowrap` keeps the 9-character "…NNNNNNNN" together
      // even when the table column is squeezed. Without it, browsers'
      // last-resort wrapping splits the ID across two lines whenever
      // the cell becomes narrower than the rendered text (which
      // happens whenever the user shrinks the dashboard's right
      // column in the GoldenLayout shell with several other columns
      // competing for width). The table's parent already has
      // overflow-auto, so the right escape hatch when there isn't
      // enough room is a horizontal scrollbar, not a mid-token break
      // - matching how relativeTime / formatters.number / badge cells
      // already behave on the same tabs.
      if (!id) return '<span class="font-mono text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">—</span>';
      const s = String(id);
      const tail = s.length > 8 ? s.slice(-8) : s;
      return `<span class="font-mono text-xs text-slate-600 dark:text-slate-300 whitespace-nowrap" title="${escape(s)}">…${escape(tail)}</span>`;
    },

    truncatePath(s) {
      if (!s) return '<span class="text-slate-500 dark:text-slate-400">—</span>';
      return `<span class="inline-block max-w-[360px] truncate align-middle font-mono text-xs text-slate-600 dark:text-slate-300" title="${escape(s)}">${escape(s)}</span>`;
    },

    number(v, opts) {
      if (v === null || v === undefined || v === '') {
        return '<span class="font-mono tabular-nums text-right text-slate-500 dark:text-slate-400">—</span>';
      }
      const n = Number(v);
      if (!isFinite(n)) {
        return '<span class="font-mono tabular-nums text-right text-slate-500 dark:text-slate-400">—</span>';
      }
      const decimals = (opts && typeof opts.decimals === 'number') ? opts.decimals : 4;
      return `<span class="font-mono tabular-nums text-right text-slate-900 dark:text-slate-100">${n.toFixed(decimals)}</span>`;
    },
  };

  function escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  }

  /* ---------- toasts -------------------------------------------- */

  let toastContainer = null;
  function ensureToastContainer() {
    if (toastContainer && document.body.contains(toastContainer)) return toastContainer;
    toastContainer = document.createElement('div');
    // Bottom-center anchor (was: top-right). The top-right slot
    // sits directly underneath each tab's right-aligned action
    // buttons (+ New job, + Eval selected, Compare in Analysis,
    // Restart, Delete, ...), so toasts confirming THOSE clicks
    // would land on top of the very button the user just pressed -
    // exactly the wrong UX. Bottom-center lands in the empty middle
    // of the TableView's pagination footer (left side has page-info
    // text only, right side has prev/next buttons; the centre is
    // empty under `ml-auto`), so the toast doesn't obscure any
    // interactive element on any tab.
    //
    // `pointer-events-none` on the container, `pointer-events-auto`
    // on each toast (set in showToast below) so clicks through the
    // empty space around toasts still reach the pagination row, but
    // clicking a toast itself dismisses it. `items-center` centers
    // toasts of different widths (240-380px) within the container's
    // own flex track so they line up vertically when stacked.
    toastContainer.className =
      'fixed bottom-4 left-1/2 -translate-x-1/2 z-[200] ' +
      'flex flex-col items-center gap-2 pointer-events-none';
    document.body.appendChild(toastContainer);
    return toastContainer;
  }

  const TOAST_BORDER_TONE = {
    success: 'border-l-emerald-500',
    warning: 'border-l-amber-500',
    error:   'border-l-red-500',
    info:    'border-l-blue-500',
  };

  function showToast(opts) {
    const { type = 'info', message = '', duration = 3500 } = opts || {};
    const container = ensureToastContainer();
    const el = document.createElement('div');
    // Initial state slides UP from below (translate-y-3) to match
    // the bottom-anchored container, then settles to translate-y-0.
    // Previously slid in from the right; that motion made sense only
    // when the container itself was top-right anchored.
    el.className =
      'pointer-events-auto min-w-[240px] max-w-[380px] ' +
      'bg-white dark:bg-slate-800 ' +
      'border border-slate-200 dark:border-slate-700 ' +
      `border-l-[3px] ${TOAST_BORDER_TONE[type] || TOAST_BORDER_TONE.info} ` +
      'rounded-lg shadow-card-md ' +
      'px-4 py-3 text-xs text-slate-900 dark:text-slate-100 ' +
      'opacity-0 translate-y-3 transition duration-200 ease-out';
    el.textContent = message;
    container.appendChild(el);
    requestAnimationFrame(() => {
      el.classList.remove('opacity-0', 'translate-y-3');
      el.classList.add('opacity-100', 'translate-y-0');
    });
    const timer = setTimeout(() => removeToast(el), duration);
    el.addEventListener('click', () => {
      clearTimeout(timer);
      removeToast(el);
    });
  }

  function removeToast(el) {
    // Mirror the show animation: slide back DOWN as we fade out.
    el.classList.remove('opacity-100', 'translate-y-0');
    el.classList.add('opacity-0', 'translate-y-3');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 500);
  }

  /* ---------- confirm dialog ------------------------------------ */

  function confirmDialog(opts) {
    const {
      title = 'Confirm',
      message = 'Are you sure?',
      confirmLabel = 'Confirm',
      cancelLabel = 'Cancel',
      confirmVariant = 'danger',  // 'danger' | 'primary'
    } = opts || {};

    return new Promise((resolve) => {
      const backdrop = document.createElement('div');
      backdrop.className =
        'fixed inset-0 z-[100] modal-blur bg-slate-900/40 dark:bg-slate-950/65 ' +
        'flex items-center justify-center p-4 ' +
        'opacity-0 transition-opacity duration-200 pointer-events-none';

      const confirmBtnClasses = confirmVariant === 'primary'
        ? 'bg-gradient-to-b from-indigo-500 to-indigo-600 hover:from-indigo-400 hover:to-indigo-500 ' +
          'text-white border border-indigo-600 shadow-card focus:shadow-focus dark:focus:shadow-focus-dark'
        : 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 hover:brightness-95 ' +
          'border border-transparent focus:shadow-focus dark:focus:shadow-focus-dark';

      backdrop.innerHTML = `
        <div role="dialog" aria-modal="true"
             class="w-full max-w-[480px] max-h-[calc(100vh-2rem)]
                    bg-white dark:bg-slate-900
                    border border-slate-200 dark:border-slate-700
                    rounded-2xl shadow-card-lg
                    flex flex-col
                    opacity-0 -translate-y-3 scale-[0.98]
                    transition duration-200">
          <div class="px-6 py-5 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
            <h2 class="m-0 text-base font-semibold tracking-tight text-slate-900 dark:text-slate-100"></h2>
            <button data-action="cancel"
                    class="px-2 py-1 text-sm rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400"
                    aria-label="Close">×</button>
          </div>
          <div class="px-6 py-5 overflow-y-auto text-sm text-slate-600 dark:text-slate-300 leading-relaxed" data-role="message"></div>
          <div class="px-6 py-4 border-t border-slate-200 dark:border-slate-700 flex gap-2 justify-end">
            <button data-action="cancel"
                    class="px-4 py-1.5 rounded-md border border-slate-300 dark:border-slate-600
                           bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700
                           text-sm font-medium text-slate-900 dark:text-slate-100
                           focus:outline-none focus:shadow-focus dark:focus:shadow-focus-dark transition"></button>
            <button data-action="confirm"
                    class="px-4 py-1.5 rounded-md text-sm font-medium transition ${confirmBtnClasses}"></button>
          </div>
        </div>`;

      backdrop.querySelector('h2').textContent = title;
      backdrop.querySelector('[data-role="message"]').textContent = message;
      const cancelBtns = backdrop.querySelectorAll('[data-action="cancel"]');
      cancelBtns.forEach((b, i) => { b.textContent = (i === 0 ? '×' : cancelLabel); });
      backdrop.querySelector('[data-action="confirm"]').textContent = confirmLabel;

      const dialog = backdrop.querySelector('[role="dialog"]');

      function close(result) {
        backdrop.classList.add('opacity-0', 'pointer-events-none');
        backdrop.classList.remove('opacity-100');
        dialog.classList.add('opacity-0', '-translate-y-3', 'scale-[0.98]');
        dialog.classList.remove('opacity-100', 'translate-y-0', 'scale-100');
        backdrop.addEventListener('transitionend', () => backdrop.remove(), { once: true });
        setTimeout(() => backdrop.remove(), 500);
        document.removeEventListener('keydown', onKey);
        resolve(result);
      }
      function onKey(ev) {
        if (ev.key === 'Escape') close(false);
        if (ev.key === 'Enter') close(true);
      }
      backdrop.addEventListener('click', (ev) => {
        if (ev.target === backdrop) close(false);
      });
      backdrop.querySelectorAll('[data-action="cancel"]').forEach((b) =>
        b.addEventListener('click', () => close(false)));
      backdrop.querySelector('[data-action="confirm"]').addEventListener('click', () => close(true));
      document.addEventListener('keydown', onKey);

      document.body.appendChild(backdrop);
      requestAnimationFrame(() => {
        backdrop.classList.remove('opacity-0', 'pointer-events-none');
        backdrop.classList.add('opacity-100', 'pointer-events-auto');
        dialog.classList.remove('opacity-0', '-translate-y-3', 'scale-[0.98]');
        dialog.classList.add('opacity-100', 'translate-y-0', 'scale-100');
      });
    });
  }

  /* ---------- typed-filter constants ---------------------------- *
   *
   * Date-range preset menu. Order matters: it's the dropdown options
   * the user sees, top-to-bottom. ms=null is the no-filter sentinel
   * ("Any time"). All other ms values are window sizes - we keep rows
   * whose sortValue (parsed as ms-since-epoch) is within that window
   * before now. Add more presets here (e.g. "Last 90 days") without
   * touching the filter logic.
   * ---------------------------------------------------------------- */
  const DATE_RANGE_PRESETS = [
    { key: 'any',     label: 'Any time',     ms: null },
    { key: '1h',      label: 'Last hour',    ms: 60 * 60 * 1000 },
    { key: '24h',     label: 'Last 24h',     ms: 24 * 60 * 60 * 1000 },
    { key: '7d',      label: 'Last 7 days',  ms: 7 * 24 * 60 * 60 * 1000 },
    { key: '30d',     label: 'Last 30 days', ms: 30 * 24 * 60 * 60 * 1000 },
    { key: '90d',     label: 'Last 90 days', ms: 90 * 24 * 60 * 60 * 1000 },
  ];

  /* ---------- TableView ----------------------------------------- */

  /**
   * Sortable / filterable / paginated table with stable selection.
   *
   * Implementation: thin wrapper around Tabulator (powered by
   * https://tabulator.info/, loaded via CDN by every tab that uses
   * this class). The public API below is intentionally identical to
   * the previous hand-rolled implementation so call sites in jobs.html,
   * models.html and leaderboard.html keep working unchanged. Tabulator
   * gives us battle-tested column resize, persistence, sort, multi-
   * filter, pagination, sticky header, and multi-row selection out of
   * the box. The wrapper handles three things Tabulator doesn't ship
   * exactly the way we want them:
   *
   *   1. **Filter type adaptation.**  Our column config exposes
   *      `filter: true` / `filter: { type: 'numeric-range' }` /
   *      `filter: { type: 'date-range' }`. The wrapper translates
   *      those into Tabulator's `headerFilter` + `headerFilterFunc`
   *      callbacks, providing custom DOM widgets for the two range
   *      types (two min/max inputs and a preset-dropdown
   *      respectively).
   *   2. **rowClass(row, ctx) hook.**  Mapped onto Tabulator's
   *      `rowFormatter` with `row.isSelected()` passed through as
   *      ctx.isSelected so callers can defer to selection styling.
   *      We also wire `rowSelected` / `rowDeselected` events to
   *      call `row.reformat()` so highlight changes take effect on
   *      every selection toggle.
   *   3. **Composite row keys.**  Some tabs (leaderboard) compute
   *      their primary key from multiple fields. Tabulator's `index`
   *      option expects a single field name, so the wrapper computes
   *      `__rcRowKey = rowKey(row)` once per row when data is set
   *      and points Tabulator at that synthetic field.
   *
   * options:
   *   columns:       Array<ColumnDef>  (required)
   *     ColumnDef = {
   *       key:        string             // field on the row object
   *       label:      string             // header text
   *       sortable?:  boolean            // default true
   *       filter?:    true | false |
   *                   { type: 'text' } |
   *                   { type: 'numeric-range' } |
   *                   { type: 'date-range' }
   *       render?:    (value, row) => string  // returns HTML
   *       sortValue?: (row) => any
   *       cellClass?: string
   *       headerClass?: string
   *     }
   *   selectable?:   boolean            // default false; adds checkbox col
   *   rowKey:        (row) => string    // stable identity; default row._id || row.id
   *   pageSize?:     number             // default 25
   *   pageSizes?:    number[]           // default [25, 50, 100, 250]
   *   emptyMessage?: string             // default 'No data.'
   *   onRowClick?:   (row) => void
   *   rowClass?:     (row, ctx) => string
   */
  class TableView {
    constructor(rootEl, options) {
      this.root = rootEl;
      this.opts = Object.assign({
        selectable: false,
        rowKey: (row) => row._id || row.id,
        pageSize: 25,
        pageSizes: [25, 50, 100, 250],
        emptyMessage: 'No data.',
        onRowClick: null,
        rowClass: null,
        columns: [],
      }, options || {});

      // Make the host fill its flex parent so Tabulator's internal
      // viewport has a defined height to scroll within. Every tab
      // places the table host inside a `flex flex-col flex-1` parent,
      // so adding `flex-1 min-h-0` here gives the table the remaining
      // vertical space without overflowing the iframe. Without this,
      // Tabulator's body would either be zero-height (no rows shown)
      // or, with auto sizing, would push the pagination footer off
      // the bottom of the iframe.
      this.root.classList.add('flex-1', 'min-h-0', 'overflow-hidden');

      // Public surface used by polling code: _rows.length === 0 is
      // how the tabs decide whether to show a full-card error vs a
      // toast on refresh failure. We maintain it in lockstep with
      // every setData call.
      this._rows = [];
      this._selectionListeners = [];
      // We trigger an explicit layout recompute the FIRST time real
      // data arrives, so columns whose content is wider than their
      // header (e.g. Model location, long EVAL summary rows) get
      // their natural widths instead of the header-only widths the
      // initial empty-table layout produces. Set false on subsequent
      // setData calls so widths don't jump as the polling refreshes
      // data every 5 s, and user-resized widths persist via the
      // built-in persistence layer.
      this._needsFirstDataLayout = true;

      if (typeof Tabulator === 'undefined') {
        // Defensive: if the CDN didn't load (offline, bad URL, etc.)
        // surface a clear message rather than throwing in the
        // constructor. The host element still gets a visible failure
        // so the user can act on it.
        this.root.innerHTML =
          '<div style="padding:1rem;color:#b91c1c;font-family:sans-serif">' +
          'Tabulator failed to load. Check the network tab + reload.' +
          '</div>';
        return;
      }

      // Map our column configs onto Tabulator's syntax. Done once at
      // construction time; the resulting array is passed straight to
      // Tabulator's `columns` option.
      const tabCols = this._buildColumns();

      // Tabulator pagination size selector wants a sorted ascending
      // array of integers; reuse opts.pageSizes verbatim.
      const persistKey =
        'roboracer:table:' +
        (typeof window !== 'undefined' && window.location ? window.location.pathname : 'unknown');

      this._tabulator = new Tabulator(this.root, {
        data: [],
        columns: tabCols,
        // Make the table fill its host container (which we sized to
        // flex-1 above). Tabulator handles the internal scrolling of
        // the body region; header stays sticky on top and pagination
        // sticks to the bottom. Without `height`, the table grows to
        // fit content and the pagination footer drifts off-screen.
        height: '100%',
        // 'fitData' lets each column size to its widest cell, which
        // mirrors what the hand-rolled TableView did. 'fitColumns'
        // would force columns to fill the table width and would
        // override resize attempts; we explicitly avoid that.
        layout: 'fitData',
        layoutColumnsOnNewData: false,
        responsiveLayout: false,
        movableColumns: false,
        resizableColumns: true,
        // Index = the synthetic __rcRowKey we stamp per row in
        // setData. Drives selection-persistence across replaceData
        // calls, plus indexing of the internal row map.
        index: '__rcRowKey',
        // Selection model: multi-row checkbox if requested. The
        // first column (rendered separately below in _buildColumns)
        // is the checkbox column.
        selectable: this.opts.selectable ? true : false,
        selectableRangeMode: 'click',
        // Pagination
        pagination: true,
        paginationMode: 'local',
        paginationSize: this.opts.pageSize,
        paginationSizeSelector: this.opts.pageSizes,
        paginationCounter: 'rows',
        // Placeholder shown when there's no data after filtering. We
        // surface the caller-supplied emptyMessage; loading/error
        // states are layered on top via class-based overlays
        // (see setLoading/setError + the .tabulator.is-loading
        // rules in styles.css).
        placeholder: this.opts.emptyMessage,
        // Persistence: remember column widths per tab. Sort + filter
        // intentionally not persisted because the tabs poll
        // continuously and tend to want fresh state on reload.
        persistenceID: persistKey,
        persistence: {
          columns: ['width'],
        },
        rowFormatter: this.opts.rowClass ? this._buildRowFormatter() : undefined,
        // Sticky header is the default in Tabulator; no opt needed.
      });

      // Hook events to forward into our listener API. Tabulator's
      // 'tableBuilt' fires once after the initial DOM is laid out;
      // we use it to gate setData() calls placed before init finishes.
      this._ready = false;
      this._pendingData = null;
      this._tabulator.on('tableBuilt', () => {
        this._ready = true;
        if (this._pendingData) {
          const d = this._pendingData;
          this._pendingData = null;
          this._tabulator.replaceData(d);
        }
      });

      if (this.opts.selectable) {
        this._tabulator.on('rowSelectionChanged', (data /*, rows */) => {
          // Tabulator gives us `data` already as an array of row
          // data objects, in selection order. Our listeners expect
          // the same shape (objects, not RowComponents), so pass
          // through directly.
          this._fireSelectionChange(data);
        });
        // Selection toggles need to re-run rowFormatter so the
        // recently-evaluated highlight switches off when a row is
        // selected (the rowClass callback short-circuits when
        // ctx.isSelected). Tabulator doesn't auto-reformat on
        // selection state changes, so trigger it explicitly.
        const reformatOnSelect = (row) => { try { row.reformat(); } catch (e) {} };
        this._tabulator.on('rowSelected',   reformatOnSelect);
        this._tabulator.on('rowDeselected', reformatOnSelect);
      }

      if (typeof this.opts.onRowClick === 'function') {
        this._tabulator.on('rowClick', (ev, row) => {
          try { this.opts.onRowClick(row.getData()); } catch (e) { console.error(e); }
        });
      }
    }

    // -------- column config translation -----------------------------

    _buildColumns() {
      const out = [];

      // Selection checkbox column first, if requested. Tabulator
      // ships a 'rowSelection' formatter that paints a checkbox in
      // the cell and a select-all checkbox in the header for free.
      // We tag BOTH the column header (.tabulator-col) and the body
      // cells (.tabulator-cell) with `rc-select-col` so styles.css
      // can reset the default `padding: 0.75rem 1rem` we apply to
      // every other column - without that reset the 32 px of left+
      // right padding plus the ~14 px checkbox overflows the 40 px
      // column, clipping the select-all checkbox in the header.
      // Cabbage-tier bug, but easy to miss because Tabulator's
      // builtin body formatter happens to wrap the checkbox in a
      // <label> that overflows visibly while the header formatter
      // doesn't.
      if (this.opts.selectable) {
        out.push({
          formatter: 'rowSelection',
          titleFormatter: 'rowSelection',
          hozAlign: 'center',
          headerHozAlign: 'center',
          headerSort: false,
          resizable: false,
          width: 40,
          cssClass: 'rc-select-col',
          headerCssClass: 'rc-select-col',
          // Selection-checkbox clicks shouldn't bubble out as a row-
          // click so we can keep onRowClick consumers honest. Tabulator
          // already stops the event for this builtin formatter.
        });
      }

      for (const c of this.opts.columns) {
        const col = {
          field: c.key,
          title: c.label,
          headerSort: c.sortable !== false,
          // resizable defaults to true; allow opt-out per column via
          // c.resizable === false if we ever need it.
          resizable: c.resizable !== false,
          // Floor for the column width: ensures the header label
          // (uppercased + letter-spaced via our theme CSS) is always
          // legible even when the cell content beneath it is much
          // narrower than the header. Tabulator's `fitData` layout
          // does take header width into account, but our header text
          // is rendered at 11px upper-case with 0.04em letter-spacing
          // which Tabulator measures inconsistently across browsers.
          // A label-length-derived minWidth pinned in CSS pixels is
          // the predictable thing. The 9 px/char + 36 px constant
          // covers the uppercase form of any label up to ~30 chars,
          // plus padding (32 px) and the sort-indicator triangle
          // (~12 px) plus a 6 px safety margin so headers never
          // ellipsize on initial layout. Users can still drag the
          // resize handle below this floor if they really want to.
          minWidth: Math.max(70, String(c.label || '').length * 9 + 36),
        };

        if (c.cellClass)   col.cssClass = c.cellClass;
        if (c.headerClass) col.headerCssClass = c.headerClass;
        if (c.hozAlign)    col.hozAlign = c.hozAlign;

        // Cell renderer. Our render callback signature is
        // (value, row) => htmlString. Tabulator's formatter
        // callback signature is (cell) => htmlString | DOMNode.
        // Bridge them transparently.
        if (typeof c.render === 'function') {
          col.formatter = (cell) => {
            try {
              return c.render(cell.getValue(), cell.getRow().getData());
            } catch (e) {
              console.error('column.render threw:', e, c.key);
              return '';
            }
          };
        }

        // Custom sort value. Our callback signature is row=>any.
        // Tabulator's sorter signature is
        // (a, b, aRow, bRow, column, dir, sorterParams). We provide
        // a wrapper that pulls sortValue off each row's data.
        if (typeof c.sortValue === 'function') {
          col.sorter = (a, b, aRow, bRow /*, column, dir */) => {
            let av, bv;
            try { av = c.sortValue(aRow.getData()); } catch (e) { av = null; }
            try { bv = c.sortValue(bRow.getData()); } catch (e) { bv = null; }
            // Mirror the previous behaviour: null sorts before
            // non-null in ascending order. Tabulator's dir handling
            // flips the result for us, so we always return as if
            // sorting ascending.
            if (av == null && bv == null) return 0;
            if (av == null) return -1;
            if (bv == null) return 1;
            if (typeof av === 'number' && typeof bv === 'number') return av - bv;
            const ad = Date.parse(av), bd = Date.parse(bv);
            if (!isNaN(ad) && !isNaN(bd)) return ad - bd;
            return String(av).localeCompare(String(bv));
          };
        }

        // Filter widget + predicate per filter type. Resolves the
        // legacy `filter: true` / `filter: false` shapes alongside
        // the new typed forms.
        const ftype = this._columnFilterType(c);
        if (ftype === 'text') {
          col.headerFilter = 'input';
          col.headerFilterPlaceholder = 'Filter…';
          col.headerFilterFunc = (headerValue, rowValue) => {
            if (headerValue == null || String(headerValue).trim() === '') return true;
            if (rowValue == null) return false;
            return String(rowValue).toLowerCase().includes(String(headerValue).trim().toLowerCase());
          };
        } else if (ftype === 'numeric-range') {
          col.headerFilter = numericRangeHeaderFilter;
          col.headerFilterLiveFilter = false;
          col.headerFilterFunc = makeNumericRangeFilterFunc(c);
        } else if (ftype === 'date-range') {
          col.headerFilter = dateRangeHeaderFilter;
          col.headerFilterLiveFilter = false;
          col.headerFilterFunc = makeDateRangeFilterFunc(c);
        }

        out.push(col);
      }
      return out;
    }

    _columnFilterType(col) {
      const f = col.filter;
      if (!f) return null;
      if (f === true) return 'text';
      if (typeof f === 'object' && f.type) return f.type;
      return 'text';
    }

    // -------- rowFormatter ------------------------------------------

    _buildRowFormatter() {
      const cb = this.opts.rowClass;
      return (row) => {
        let rowData;
        try { rowData = row.getData(); } catch (e) { return; }
        let isSelected = false;
        try { isSelected = !!row.isSelected(); } catch (e) {}
        let cls = '';
        try { cls = cb(rowData, { isSelected }) || ''; } catch (e) {
          console.error('rowClass callback threw:', e);
        }
        const el = row.getElement();
        // Track which classes WE added so we can remove them cleanly
        // before applying the next set. Without this, classes leak
        // across reformats and an unselect doesn't remove the row-
        // eval-fresh class we put there earlier.
        const tracked = el.__rcAppliedClasses || [];
        for (const c of tracked) el.classList.remove(c);
        const fresh = String(cls).trim();
        if (fresh) {
          const parts = fresh.split(/\s+/);
          for (const c of parts) el.classList.add(c);
          el.__rcAppliedClasses = parts;
        } else {
          el.__rcAppliedClasses = [];
        }
      };
    }

    // -------- public API --------------------------------------------

    setData(rows /*, opts */) {
      const arr = Array.isArray(rows) ? rows : [];
      this._rows = arr.slice();
      // Stamp the synthetic primary key onto each row so Tabulator's
      // selection persists across data updates. We always compute it
      // (cheap) so callers can pass anything through opts.rowKey.
      const keyFn = this.opts.rowKey;
      const decorated = arr.map((r) => {
        // Avoid mutating caller's object unless we have to. Spread
        // creates a shallow copy so the original row object the
        // caller keeps referencing isn't tagged with our key.
        const k = keyFn ? keyFn(r) : (r && (r._id || r.id));
        return Object.assign({ __rcRowKey: String(k) }, r);
      });
      if (!this._tabulator) return;
      // If tableBuilt hasn't fired yet, queue the data; Tabulator's
      // replaceData throws if called pre-init.
      if (!this._ready) {
        this._pendingData = decorated;
        return;
      }
      // Clear any error/loading overlay set by the previous render
      // - new data implies the underlying request succeeded.
      this.setLoading(false);
      this.setError(null);
      // Tabulator's replaceData returns a promise; once data is in,
      // the very first non-empty load triggers a full redraw so the
      // fitData layout algorithm picks up real cell content widths.
      // Without this, the table stays sized for an empty grid
      // (header-width-only columns) and content that's WIDER than
      // the header is ellipsized until the user manually resizes.
      // Subsequent setData calls skip the redraw so user-resized
      // columns + steady-state widths don't jump on every poll tick.
      const replaceResult = this._tabulator.replaceData(decorated);
      if (this._needsFirstDataLayout && decorated.length > 0) {
        this._needsFirstDataLayout = false;
        const doRelayout = () => {
          try {
            this._tabulator.redraw(true);
          } catch (err) {
            // Tabulator can throw if called pre-init or post-
            // destroy; in both cases the user sees default widths
            // and a manual resize still works.
          }
        };
        if (replaceResult && typeof replaceResult.then === 'function') {
          replaceResult.then(doRelayout, doRelayout);
        } else {
          // Older Tabulator versions return undefined; fall back to
          // a microtask so the DOM update commits before we measure.
          Promise.resolve().then(doRelayout);
        }
      }
      // Selection events don't fire automatically on data refresh,
      // even though Tabulator preserves selection by __rcRowKey. Re-
      // notify listeners so toolbar button state stays accurate.
      this._fireSelectionChange();
    }

    setLoading(b) {
      if (!this._tabulator) return;
      const el = this._tabulator.element;
      if (b) {
        el.dataset.overlay = 'Loading…';
        el.classList.add('is-loading');
      } else {
        el.classList.remove('is-loading');
      }
    }

    setError(msg) {
      if (!this._tabulator) return;
      const el = this._tabulator.element;
      if (msg) {
        el.dataset.overlay = String(msg);
        el.classList.add('has-error');
        el.classList.remove('is-loading');
      } else {
        el.classList.remove('has-error');
      }
    }

    onSelectionChange(fn) {
      if (typeof fn === 'function') this._selectionListeners.push(fn);
    }

    getSelectedRows() {
      if (!this._tabulator) return [];
      try {
        return this._tabulator.getSelectedData() || [];
      } catch (e) {
        return [];
      }
    }

    clearSelection() {
      if (!this._tabulator) return;
      try { this._tabulator.deselectRow(); } catch (e) {}
      this._fireSelectionChange();
    }

    // Reapply Tabulator-side filters from this wrapper instance, useful
    // for the "Only evaluated" toggle on the Models tab. The toggle
    // can call `table.applyExternalFilter(...)` instead of touching
    // the column-level header filters directly. Not used internally
    // by the wrapper itself, but exposed for the consuming tabs.
    applyExternalFilter(field, type, value) {
      if (!this._tabulator) return;
      this._tabulator.setFilter(field, type, value);
    }
    clearExternalFilter() {
      if (!this._tabulator) return;
      try { this._tabulator.clearFilter(false); } catch (e) {}
    }

    // -------- internals ---------------------------------------------

    _fireSelectionChange(prefetched) {
      const sel = prefetched != null ? prefetched : this.getSelectedRows();
      for (const fn of this._selectionListeners) {
        try { fn(sel); } catch (e) { console.error(e); }
      }
    }
  }

  // ---------- custom Tabulator headerFilter widgets ----------------
  //
  // Tabulator's `headerFilter` accepts either a built-in widget name
  // ('input', 'select', 'number', etc.) or a factory function. The
  // factory function signature is:
  //
  //   (cell, onRendered, success, cancel, editorParams) => DOMNode
  //
  // It builds the editor DOM, optionally registers cleanup callbacks
  // with `onRendered`, and calls `success(value)` to commit the
  // filter value. We pair each factory with a `headerFilterFunc`
  // (built per-column inside _buildColumns) that interprets the value
  // shape against each row.

  function numericRangeHeaderFilter(cell, onRendered, success /*, cancel, editorParams */) {
    // Two compact number inputs. Calling success({min, max}) on any
    // change triggers Tabulator to re-filter. Empty string in either
    // box means "unbounded on that side"; the matching
    // headerFilterFunc clamps accordingly.
    const wrap = document.createElement('div');
    wrap.className = 'rc-numeric-range-filter';
    const mk = (placeholder) => {
      const i = document.createElement('input');
      i.type = 'number';
      i.placeholder = placeholder;
      i.inputMode = 'decimal';
      // Stop arrow keys from being interpreted as Tabulator
      // navigation while the input has focus.
      i.addEventListener('keydown', (ev) => ev.stopPropagation());
      return i;
    };
    const minInput = mk('min');
    const maxInput = mk('max');
    wrap.appendChild(minInput);
    wrap.appendChild(maxInput);
    const onChange = () => {
      success({
        min: minInput.value === '' ? null : minInput.value,
        max: maxInput.value === '' ? null : maxInput.value,
      });
    };
    minInput.addEventListener('input', onChange);
    maxInput.addEventListener('input', onChange);
    return wrap;
  }

  function makeNumericRangeFilterFunc(col) {
    return (headerValue, rowValue, rowData /*, filterParams */) => {
      if (!headerValue) return true;
      const lo = headerValue.min != null ? Number(headerValue.min) : null;
      const hi = headerValue.max != null ? Number(headerValue.max) : null;
      const loActive = Number.isFinite(lo);
      const hiActive = Number.isFinite(hi);
      if (!loActive && !hiActive) return true;
      const valFn = col.sortValue || ((r) => r[col.key]);
      const v = Number(valFn(rowData));
      if (!Number.isFinite(v)) return false;
      if (loActive && v < lo) return false;
      if (hiActive && v > hi) return false;
      return true;
    };
  }

  function dateRangeHeaderFilter(cell, onRendered, success /*, cancel, editorParams */) {
    // Preset dropdown driven by DATE_RANGE_PRESETS at the top of this
    // module. Calling success(preset.key) triggers Tabulator to
    // re-filter; the paired headerFilterFunc interprets the key.
    const sel = document.createElement('select');
    sel.className = 'rc-date-range-filter';
    sel.addEventListener('keydown', (ev) => ev.stopPropagation());
    for (const p of DATE_RANGE_PRESETS) {
      const o = document.createElement('option');
      o.value = p.key;
      o.textContent = p.label;
      sel.appendChild(o);
    }
    sel.addEventListener('change', () => success(sel.value));
    return sel;
  }

  function makeDateRangeFilterFunc(col) {
    return (headerValue, rowValue, rowData /*, filterParams */) => {
      if (!headerValue || headerValue === 'any') return true;
      const preset = DATE_RANGE_PRESETS.find((p) => p.key === headerValue);
      if (!preset || preset.ms == null) return true;
      const cutoff = Date.now() - preset.ms;
      const valFn = col.sortValue || ((r) => r[col.key]);
      const raw = valFn(rowData);
      const t = typeof raw === 'number'
        ? raw
        : (raw == null ? NaN : Date.parse(String(raw)));
      return Number.isFinite(t) && t >= cutoff;
    };
  }


  /* ---------- FreshnessIndicator -------------------------------- */

  /**
   * "● updated Ns ago" pill that ticks every second and ramps its dot
   * color from emerald to amber to red as the tracked poll's age
   * increases. Used by jobs.html / models.html / leaderboard.html to
   * give the user a clear "is the data still flowing?" signal in the
   * toolbar.
   *
   * Each tab owns its own instance because each polls a different
   * endpoint at potentially different cadences. Wire-up:
   *
   *   <span data-role="freshness" ...>
   *     <span data-role="freshness-dot" ...></span>
   *     <span data-role="freshness-text"></span>
   *   </span>
   *
   *   const freshness = new RoboracerUI.FreshnessIndicator(
   *     document.querySelector('[data-role="freshness-dot"]'),
   *     document.querySelector('[data-role="freshness-text"]'));
   *
   *   // In your pollOnce(...):
   *   const res = await fetch(...);
   *   if (res.ok) freshness.markSuccess();
   *
   *   // On page unload:
   *   window.addEventListener('beforeunload', () => freshness.destroy());
   *
   * The "success" semantics include NO_CHANGES short-circuit responses
   * - the server reaching MongoDB and reporting "nothing new" is
   * still proof the polling path is healthy.
   */
  class FreshnessIndicator {
    constructor(dotEl, textEl, opts) {
      this.dotEl = dotEl;
      this.textEl = textEl;
      this.opts = Object.assign({
        freshMs: 8000,    // < this = emerald (within one 5s poll cycle + slop)
        staleMs: 30000,   // < this = amber, ≥ this = red
        tickMs: 1000,
      }, opts || {});
      this.lastSuccessAt = null;
      this._interval = setInterval(() => this._render(), this.opts.tickMs);
      this._render();
    }

    /** Call after each successful poll (including NO_CHANGES). */
    markSuccess() {
      this.lastSuccessAt = Date.now();
      this._render();
    }

    /** Stop the 1-second ticker. Call on page unload. */
    destroy() {
      clearInterval(this._interval);
    }

    _formatTimestamp(ts) {
      // "May 14, 6:31 PM" style. Uses the browser's locale so users
      // outside en-US see their own conventions (e.g. "14 May, 18:31"
      // in en-GB). The dot color still conveys freshness, so the text
      // is free to be an absolute timestamp rather than relative.
      const d = new Date(ts);
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    }

    _render() {
      if (!this.dotEl || !this.textEl) return;
      if (this.lastSuccessAt === null) {
        this.dotEl.className =
          'inline-block w-1.5 h-1.5 rounded-full bg-slate-500 animate-pulse';
        this.textEl.textContent = 'connecting…';
        return;
      }
      const elapsed = Date.now() - this.lastSuccessAt;
      let dotColor;
      if (elapsed < this.opts.freshMs)      dotColor = 'bg-emerald-500';
      else if (elapsed < this.opts.staleMs) dotColor = 'bg-amber-500';
      else                                  dotColor = 'bg-red-500';
      this.dotEl.className = 'inline-block w-1.5 h-1.5 rounded-full ' + dotColor;
      // Absolute timestamp of the last successful poll. The dot color
      // (which changes every second as `elapsed` grows) is what
      // signals "stale" — the text gives the user a definite reference
      // point to compare against the wall-clock.
      this.textEl.textContent = 'updated ' + this._formatTimestamp(this.lastSuccessAt);
    }
  }

  /* ---------- exports ------------------------------------------- */

  global.RoboracerUI = {
    formatters,
    showToast,
    confirmDialog,
    TableView,
    FreshnessIndicator,
  };
})(window);
