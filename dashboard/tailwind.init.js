/* ---------------------------------------------------------------- *
 * Tailwind config for the rl-roboracer dashboard.
 *
 * Loaded *after* the Tailwind Play CDN script in each tab's <head>:
 *
 *   <script src="https://cdn.tailwindcss.com"></script>
 *   <script src="/tailwind.init.js"></script>
 *
 * Order matters - this file relies on `window.tailwind` being defined
 * by the Play CDN script. Browsers execute <script src=...> tags in
 * source order so the dependency is already loaded by the time we run.
 *
 * Centralised here so all three tabs (jobs / models / leaderboard)
 * share the same theme: a slate-neutral palette + indigo-500 accent,
 * Inter sans + JetBrains Mono mono, dark mode tracking the OS via
 * prefers-color-scheme.
 * ---------------------------------------------------------------- */

(function () {
  if (!window.tailwind) {
    console.warn('tailwind.init.js: window.tailwind is undefined. ' +
                 'Ensure the Play CDN script is loaded before this file.');
    return;
  }

  window.tailwind.config = {
    /* darkMode 'class' activates `dark:` variants whenever a `.dark`
     * ancestor is present (we set `class="dark"` on <html> in each
     * tab to force always-dark). Switching back to `'media'` here +
     * removing the `dark` class from each HTML's <html> would
     * restore OS-tracking. */
    darkMode: 'class',

    theme: {
      extend: {
        fontFamily: {
          /* Inter loaded via Google Fonts in styles.css. The system-
           * font fallback chain mirrors what we used pre-Tailwind so
           * the dashboard still renders cleanly while Inter loads. */
          sans: [
            'Inter', '"SF Pro Text"', '-apple-system', 'BlinkMacSystemFont',
            '"Segoe UI Variable"', '"Segoe UI"', 'system-ui', 'Roboto',
            '"Helvetica Neue"', 'Arial', 'sans-serif',
          ],
          mono: [
            '"JetBrains Mono"', '"SF Mono"', 'Menlo', 'Consolas',
            '"Liberation Mono"', 'monospace',
          ],
        },
        boxShadow: {
          /* Soft 3px focus ring around interactive elements. Mirrors
           * the var(--shadow-focus) we used pre-Tailwind. The dark
           * variant uses a brighter indigo so it remains visible
           * against dark surfaces. */
          'focus':       '0 0 0 3px rgba(99, 102, 241, 0.32)',
          'focus-dark':  '0 0 0 3px rgba(129, 140, 248, 0.42)',
          /* Card-style elevation at 3 levels: subtle / standard /
           * floating. Multi-layer to feel ambient rather than flat. */
          'card':        '0 1px 2px rgba(15, 23, 42, 0.05), 0 1px 1px rgba(15, 23, 42, 0.03)',
          'card-md':     '0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.04)',
          'card-lg':     '0 12px 24px -8px rgba(15, 23, 42, 0.18), 0 4px 8px -4px rgba(15, 23, 42, 0.08)',
        },
        animation: {
          /* Slightly snappier than Tailwind's default `spin` (1s) - the
           * spinner sits on a loading state and feels more responsive
           * at 0.8s. */
          'spin-fast': 'spin 0.8s cubic-bezier(0.5, 0, 0.5, 1) infinite',
          'fade-in':   'fadeIn 200ms cubic-bezier(0.4, 0, 0.2, 1)',
          'pop-in':    'popIn 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        },
        keyframes: {
          fadeIn: {
            '0%':   { opacity: '0' },
            '100%': { opacity: '1' },
          },
          popIn: {
            '0%':   { opacity: '0', transform: 'translateY(12px) scale(0.98)' },
            '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
          },
        },
      },
    },
  };
})();
