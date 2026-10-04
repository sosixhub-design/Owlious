/*
  OWLIOUS API — Cloudflare Worker
  ================================
  This is the shared "brain" that connects index.html (the player site) and
  admin.html (the admin uploader). It stores every pack/effect as one JSON
  list inside a Cloudflare KV namespace, and both HTML files talk to it
  through store.js.

  WHAT EACH PART DOES
  --------------------
  - GET  /api/items                 → anyone can read the full list (this is
                                       what makes packs show up on the site)
  - POST /api/items                 → admin-only: add or update one item
  - DELETE /api/items/:id           → admin-only: delete one item forever
  - POST /api/items/:id/view        → anyone: +1 view count
  - POST /api/items/:id/download    → anyone: +1 download count
  - POST /api/items/:id/share       → anyone: +1 share count

  "Admin-only" is enforced HERE, on the server, by checking a secret key
  sent in the `X-Admin-Key` request header against env.ADMIN_SECRET (a
  value you set in Cloudflare, never written into this file). This is the
  real security — it replaces the old "anyone who reads admin.html's
  source can see the key" problem, because the real check now happens on
  Cloudflare's servers where visitors can't see it.

  SETUP
  -----
  You don't need to understand this file to use it — the setup guide
  (CLOUDFLARE_SETUP_GUIDE.md) walks through deploying it step by step from
  your phone. In short: create a KV namespace, bind it to this Worker as
  ITEMS_KV, set a secret called ADMIN_SECRET, deploy.
*/

const KV_KEY = 'owlious_items_v1';

// Allow requests from any origin. If you want to lock this down later to
// only your own Pages domain, replace '*' with your site's URL, e.g.
// 'https://owlious.pages.dev'.
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, X-Admin-Key'
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS_HEADERS }
  });
}

async function getItems(env) {
  const raw = await env.ITEMS_KV.get(KV_KEY);
  return raw ? JSON.parse(raw) : [];
}

async function saveItems(env, items) {
  await env.ITEMS_KV.put(KV_KEY, JSON.stringify(items));
}

// The one and only place that decides whether a request is really from the
// admin. Every write operation (save, delete) must pass this check.
function isAdmin(request, env) {
  const key = request.headers.get('X-Admin-Key') || '';
  return !!env.ADMIN_SECRET && key === env.ADMIN_SECRET;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Browsers send an OPTIONS "preflight" request before POST/DELETE —
    // answer it immediately with the CORS headers so the real request
    // is allowed through.
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // GET /api/items — public, powers the whole site's Browse/Packs pages.
    if (path === '/api/items' && request.method === 'GET') {
      const items = await getItems(env);
      return json(items);
    }

    // POST /api/items — admin-only, add or update one item (upsert by id).
    if (path === '/api/items' && request.method === 'POST') {
      if (!isAdmin(request, env)) return json({ error: 'Not authorized.' }, 401);
      let item;
      try { item = await request.json(); } catch (_) { return json({ error: 'Bad JSON.' }, 400); }
      if (!item || typeof item !== 'object' || item.id == null) {
        return json({ error: 'Item must include an id.' }, 400);
      }
      const items = await getItems(env);
      const idx = items.findIndex(it => it.id === item.id);
      if (idx >= 0) items[idx] = item; else items.push(item);
      await saveItems(env, items);
      return json(item);
    }

    // DELETE /api/items/:id — admin-only, permanent delete.
    const delMatch = path.match(/^\/api\/items\/([^/]+)$/);
    if (delMatch && request.method === 'DELETE') {
      if (!isAdmin(request, env)) return json({ error: 'Not authorized.' }, 401);
      const id = delMatch[1];
      const items = await getItems(env);
      const filtered = items.filter(it => String(it.id) !== String(id));
      await saveItems(env, filtered);
      return json({ ok: true });
    }

    // POST /api/items/:id/view | /download | /share — public counters.
    const counterMatch = path.match(/^\/api\/items\/([^/]+)\/(view|download|share)$/);
    if (counterMatch && request.method === 'POST') {
      const id = counterMatch[1];
      const field = counterMatch[2] === 'view' ? 'views'
        : counterMatch[2] === 'download' ? 'downloads' : 'shares';
      const items = await getItems(env);
      const item = items.find(it => String(it.id) === String(id));
      if (!item) return json({ error: 'Not found.' }, 404);
      item[field] = (item[field] || 0) + 1;
      await saveItems(env, items);
      return json({ [field]: item[field] });
    }

    return json({ error: 'Not found.' }, 404);
  }
};
