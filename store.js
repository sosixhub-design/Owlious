/*
  OWLIOUS STORE — shared data client
  ===================================
  Both index.html and admin.html load this file. It talks to your
  Cloudflare Worker API so that packs uploaded in admin.html actually show
  up for every visitor on index.html — that's the "connection" between the
  two files.

  SETUP: after you deploy the Worker (see CLOUDFLARE_SETUP_GUIDE.md), copy
  its URL and paste it below as API_BASE. That's the only edit this file
  needs.
*/

// PASTE YOUR WORKER URL HERE, with no trailing slash, e.g.:
// const API_BASE = 'https://owlious-api.yourname.workers.dev';
const API_BASE = 'PASTE_YOUR_WORKER_URL_HERE';

(function () {
  const configured = API_BASE && !API_BASE.startsWith('PASTE_');

  // The admin key is never stored in this file. admin.html calls
  // PhantomStore.setAdminKey(key) right after a successful login, and it's
  // kept only in memory for that page session — then sent as a header on
  // every write request, where the Worker is the one that actually checks
  // it against the real secret.
  let adminKey = '';

  async function apiGet(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error('Request failed (' + res.status + ')');
    return res.json();
  }

  async function apiSend(path, method, body) {
    const res = await fetch(API_BASE + path, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-Admin-Key': adminKey
      },
      body: body !== undefined ? JSON.stringify(body) : undefined
    });
    if (!res.ok) {
      const msg = res.status === 401 ? 'Admin key rejected by the server.' : 'Request failed (' + res.status + ')';
      throw new Error(msg);
    }
    return res.json();
  }

  // Local fallback cache, used only if the Worker can't be reached (e.g. no
  // signal, or API_BASE hasn't been set up yet), so the site never shows a
  // blank error page — it just shows the last-known list.
  const CACHE_KEY = 'owliousItemsCache';
  function readCache() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || []; } catch (_) { return []; }
  }
  function writeCache(items) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(items)); } catch (_) {}
  }

  window.PhantomStore = {
    // Lets admin.html hand over the key the person just typed, so writes
    // can be authenticated. Never persisted to disk.
    setAdminKey(key) { adminKey = key || ''; },

    async getItems() {
      if (!configured) return readCache();
      try {
        const items = await apiGet('/api/items');
        writeCache(items);
        return items;
      } catch (_) {
        // Offline or the Worker is down — fall back to whatever we last saw.
        return readCache();
      }
    },

    async saveItem(item) {
      if (!configured) throw new Error('Connect store.js to your Worker first (see CLOUDFLARE_SETUP_GUIDE.md).');
      return apiSend('/api/items', 'POST', item);
    },

    async deleteItem(id) {
      if (!configured) throw new Error('Connect store.js to your Worker first (see CLOUDFLARE_SETUP_GUIDE.md).');
      return apiSend('/api/items/' + encodeURIComponent(id), 'DELETE');
    },

    // Public counters — any visitor can trigger these, no admin key needed.
    async registerView(id) {
      if (!configured) return;
      try { await apiSend('/api/items/' + encodeURIComponent(id) + '/view', 'POST'); } catch (_) {}
    },
    async registerDownload(id) {
      if (!configured) return;
      try { await apiSend('/api/items/' + encodeURIComponent(id) + '/download', 'POST'); } catch (_) {}
    },
    async registerShare(id) {
      if (!configured) return;
      try { await apiSend('/api/items/' + encodeURIComponent(id) + '/share', 'POST'); } catch (_) {}
    }
  };
})();
