// Message search page: progressive enhancement over server-rendered HTML.
// No framework and no inline script (the Content Security Policy allows
// only this file). Every request is same-origin and carries the session
// cookie; state-changing requests also carry the CSRF token.
"use strict";

(function () {
  const csrf = () => {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") || "" : "";
  };

  async function fetchText(url) {
    const response = await fetch(url, { credentials: "same-origin", headers: { Accept: "text/html" } });
    if (response.status === 401) { window.location.assign("/login"); throw new Error("signed out"); }
    if (!response.ok) throw new Error("HTTP " + response.status);
    return response.text();
  }

  async function fetchJSON(url, options) {
    const response = await fetch(url, Object.assign({ credentials: "same-origin" }, options || {}));
    if (response.status === 401) { window.location.assign("/login"); throw new Error("signed out"); }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || ("HTTP " + response.status));
    return body;
  }

  function fragment(html) {
    const template = document.createElement("template");
    template.innerHTML = html;
    return template.content;
  }

  // ---------------------------------------------------------------- place in the results
  // The results page keeps its place in its own history entry
  // (history.replaceState): how many pages of conversations were loaded,
  // which conversations were expanded, and the scroll position. Back from
  // a conversation reloads the page (every response is no-store), and this
  // state lets it load the same pages again and return to the same spot.
  // Nothing is written to browser storage.

  const PLACE = "imsgResults";

  function savedPlace() {
    const state = history.state;
    return state && typeof state === "object" && state[PLACE] ? state[PLACE] : null;
  }

  function savePlace(patch) {
    try {
      const state = Object.assign({}, history.state || {});
      state[PLACE] = Object.assign({ pages: 1, y: 0, expanded: {} }, state[PLACE] || {}, patch);
      history.replaceState(state, "");
    } catch (_error) { /* keeping the place is a convenience */ }
  }

  function topOfScreen() {
    // The first hit (or conversation heading) still visible under the
    // sticky search bar, and how far its top sits from the top of the
    // window. Restoring to it keeps the place when something above it
    // changes height after the restore, such as photos finishing loading.
    const bar = document.querySelector(".topbar");
    const below = bar ? bar.getBoundingClientRect().bottom : 0;
    for (const el of document.querySelectorAll(".thread-result .thread-head, .thread-result .hit")) {
      const box = el.getBoundingClientRect();
      if (box.bottom > below) {
        const section = el.closest(".thread-result");
        return {
          anchor: el.classList.contains("hit") ? "hit:" + el.dataset.hit : "thread:" + (section ? section.dataset.thread : ""),
          offset: Math.round(box.top),
        };
      }
    }
    return { anchor: null, offset: 0 };
  }

  function findAnchor(results, anchor) {
    if (!anchor) return null;
    const [kind, key] = [anchor.slice(0, anchor.indexOf(":")), anchor.slice(anchor.indexOf(":") + 1)];
    if (kind === "hit") {
      return Array.from(results.querySelectorAll(".hit")).find((h) => h.dataset.hit === key) || null;
    }
    const section = Array.from(results.querySelectorAll(".thread-result")).find((s) => s.dataset.thread === key);
    return section ? section.querySelector(".thread-head") : null;
  }

  function rememberExpanded(threadKey, shown) {
    if (!threadKey) return;
    const expanded = Object.assign({}, (savedPlace() || {}).expanded || {});
    expanded[threadKey] = shown;
    savePlace({ expanded: expanded });
  }

  // ---------------------------------------------------------------- results

  let pagesLoaded = 1;
  let userMoved = false;
  window.addEventListener("scroll", () => { if (window.scrollY > 150) userMoved = true; }, { passive: true });

  const pageObserver = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) loadNextPage(entry.target);
        }
      }, { rootMargin: "800px 0px" })
    : null;

  function watchSentinels(root) {
    if (!pageObserver) return;
    root.querySelectorAll(".sentinel[data-next]").forEach((el) => pageObserver.observe(el));
  }

  async function loadNextPage(sentinel) {
    if (sentinel.dataset.loading) return;
    sentinel.dataset.loading = "1";
    if (pageObserver) pageObserver.unobserve(sentinel);
    try {
      const html = await fetchText(sentinel.dataset.next);
      const content = fragment(html);
      const parent = sentinel.parentNode;
      parent.replaceChild(content, sentinel);
      pagesLoaded += 1;
      savePlace({ pages: pagesLoaded });
      watchSentinels(parent);
    } catch (error) {
      delete sentinel.dataset.loading;
      sentinel.textContent = "Could not load more: " + error.message;
    }
  }

  async function replaceResults(url) {
    const results = document.getElementById("results");
    if (!results) return;
    const html = await fetchText(url);
    results.replaceChildren(fragment(html));
    pagesLoaded = 1;
    savePlace({ pages: 1, expanded: {} });
    watchSentinels(results);
  }

  function liveSection(key) {
    return Array.from(document.querySelectorAll(".thread-result")).find((s) => s.dataset.thread === key) || null;
  }

  async function expandThread(key, until) {
    // Load a conversation's hits 50 at a time until `until` are shown. The
    // first step replaces the conversation's section, so look it up again
    // each time.
    let section = liveSection(key);
    let button = section ? section.querySelector(".more-hits") : null;
    while (button && section && section.querySelectorAll(".hit").length < until) {
      button = await loadMoreHits(button);
      section = liveSection(key);
    }
  }

  async function loadMoreHits(button) {
    // One step of "show all hits": returns the next step's button, if any.
    button.disabled = true;
    const section = button.closest(".thread-result");
    const html = await fetchText(button.dataset.url);
    const content = fragment(html);
    let next = null;
    if (button.dataset.append && section) {
      next = content.querySelector(".more-hits");
      if (next) next.remove();
      section.querySelector(".hits").appendChild(content);
      if (next) button.replaceWith(next); else button.remove();
    } else if (section) {
      const replacement = content.querySelector(".thread-result");
      section.replaceWith(content);
      next = replacement ? replacement.querySelector(".more-hits") : null;
    }
    const key = section ? section.dataset.thread : null;
    const holder = key ? liveSection(key) : null;
    if (holder) rememberExpanded(key, holder.querySelectorAll(".hit").length);
    return next;
  }

  async function restorePlace(results) {
    const place = savedPlace();
    if (!place) return;
    userMoved = true; // never swap the list under a restored position
    for (let page = 1; page < (place.pages || 1); page++) {
      const sentinel = results.querySelector(".sentinel[data-next]");
      if (!sentinel) break;
      await loadNextPage(sentinel);
    }
    for (const [key, shown] of Object.entries(place.expanded || {})) {
      try { await expandThread(key, shown); } catch (_error) { /* shown as loaded */ }
    }
    const anchor = findAnchor(results, place.anchor);
    if (anchor) {
      window.scrollTo(0, anchor.getBoundingClientRect().top + window.scrollY - (place.offset || 0));
    } else {
      window.scrollTo(0, place.y || 0);
    }
  }

  async function runSemantic() {
    const results = document.getElementById("results");
    if (!results || !results.dataset.semanticUrl) return;
    let answer;
    try {
      answer = await fetchJSON(results.dataset.semanticUrl);
    } catch (error) {
      const status = document.getElementById("status");
      const pending = status && status.querySelector(".semantic.pending");
      if (pending) { pending.textContent = "semantic search failed (" + error.message + ")"; pending.className = "semantic unavailable"; }
      return;
    }
    const status = document.getElementById("status");
    if (status && answer.status_html) status.replaceWith(fragment(answer.status_html));
    const changed = answer.state === "done" && (answer.added_hits > 0 || answer.reranked);
    if (!changed && !answer.reranked) return;
    if (!userMoved && pagesLoaded === 1) {
      await replaceResults(answer.page1_url);
      return;
    }
    const banner = document.getElementById("semantic-banner");
    if (!banner) return;
    const text = answer.reranked
      ? "Reranked the best matches."
      : "Semantic search added " + answer.added_hits + " hit" + (answer.added_hits === 1 ? "" : "s") +
        (answer.added_threads ? " in " + answer.added_threads + " new conversation" + (answer.added_threads === 1 ? "" : "s") : "") + ".";
    banner.replaceChildren(document.createTextNode(text + " "));
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Show updated results";
    button.addEventListener("click", async () => {
      banner.hidden = true;
      await replaceResults(answer.page1_url);
      window.scrollTo({ top: 0 });
    });
    banner.appendChild(button);
    banner.hidden = false;
  }

  // ---------------------------------------------------------------- labels

  async function onLabelClick(button) {
    const controls = button.closest(".label-controls");
    const results = document.getElementById("results");
    const query = results ? results.dataset.query : null;
    if (!controls || !query) return;
    const grade = button.classList.contains("on") ? null : Number(button.dataset.grade);
    const buttons = controls.querySelectorAll(".label-btn");
    buttons.forEach((b) => { b.disabled = true; });
    try {
      const answer = await fetchJSON("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf() },
        body: JSON.stringify({ q: query, kind: controls.dataset.kind, key: controls.dataset.key, grade: grade }),
      });
      buttons.forEach((b) => {
        const on = answer.grade !== null && Number(b.dataset.grade) === answer.grade;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      const counts = answer.counts || {};
      const set = (selector, value) => { const el = document.querySelector(selector); if (el) el.textContent = String(value); };
      set(".label-count .n-total", counts.total);
      set(".label-count .n-rel", counts.relevant);
      set(".label-count .n-notrel", counts.not_relevant);
    } catch (error) {
      button.title = "Label not saved: " + error.message;
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  // ---------------------------------------------------------------- copy citation

  async function copyText(text) {
    // The page is usually plain HTTP on the local network, where the
    // Clipboard API does not exist; a selected text area and the copy
    // command work there.
    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
      try { await navigator.clipboard.writeText(text); return true; } catch (_error) { /* try the other way */ }
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.className = "copy-buffer";
    document.body.appendChild(area);
    area.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch (_error) { copied = false; }
    area.remove();
    return copied;
  }

  async function onCiteClick(button) {
    const text = button.dataset.cite || "";
    if (await copyText(text)) {
      const label = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = label; }, 1500);
      return;
    }
    // Nothing could copy: show the citation selected, to copy by hand.
    let box = button.parentNode.querySelector(".cite-box");
    if (!box) {
      box = document.createElement("input");
      box.type = "text";
      box.readOnly = true;
      box.className = "cite-box";
      box.setAttribute("aria-label", "Citation");
      button.after(box);
    }
    box.value = text;
    box.focus();
    box.select();
  }

  // ---------------------------------------------------------------- grading mode

  async function onGradeClick(button) {
    const controls = button.closest(".grade-controls");
    if (!controls) return;
    const grade = button.classList.contains("on") ? null : Number(button.dataset.grade);
    const buttons = controls.querySelectorAll(".grade-btn");
    buttons.forEach((b) => { b.disabled = true; });
    try {
      const answer = await fetchJSON("/api/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf() },
        body: JSON.stringify({ list: Number(controls.dataset.list), anchor: controls.dataset.anchor, grade: grade }),
      });
      buttons.forEach((b) => {
        const on = answer.grade !== null && Number(b.dataset.grade) === answer.grade;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      const page = document.querySelector(".grading");
      const counter = document.querySelector(".grade-progress .n-graded");
      if (page && counter) {
        counter.textContent = String(Number(page.dataset.positions) > 20 ? answer.graded_extra : answer.graded);
      }
    } catch (error) {
      button.title = "Grade not saved: " + error.message;
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  // ---------------------------------------------------------------- details panel

  async function onDetailsClick(button) {
    const row = button.closest(".msg-actions");
    if (!row) return;
    const open = row.nextElementSibling && row.nextElementSibling.classList.contains("details")
      ? row.nextElementSibling : null;
    if (open) {
      open.remove();
      button.setAttribute("aria-expanded", "false");
      return;
    }
    button.disabled = true;
    try {
      const html = await fetchText(button.dataset.url);
      row.after(fragment(html));
      button.setAttribute("aria-expanded", "true");
    } catch (error) {
      button.title = "Could not load the details: " + error.message;
    } finally {
      button.disabled = false;
    }
  }

  // ---------------------------------------------------------------- misc clicks

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const label = target.closest(".label-btn");
    if (label) { event.preventDefault(); onLabelClick(label); return; }
    const cite = target.closest(".cite-btn");
    if (cite) { event.preventDefault(); onCiteClick(cite); return; }
    const grade = target.closest(".grade-btn");
    if (grade) { event.preventDefault(); onGradeClick(grade); return; }
    const details = target.closest(".details-btn");
    if (details) { event.preventDefault(); onDetailsClick(details); return; }
    const more = target.closest(".more-hits");
    if (more) {
      event.preventDefault();
      loadMoreHits(more).catch((error) => { more.disabled = false; more.textContent = "Could not load: " + error.message; });
      return;
    }
    const back = target.closest("a.back");
    if (back && document.referrer) {
      // "Results" returns to the results page the owner came from, with its
      // place kept, rather than opening a fresh copy of it.
      try {
        const came = new URL(document.referrer);
        if (came.origin === window.location.origin && came.pathname === "/search" && history.length > 1) {
          event.preventDefault();
          history.back();
          return;
        }
      } catch (_error) { /* follow the link */ }
    }
    const pdf = target.closest(".pdf-toggle");
    if (pdf) {
      event.preventDefault();
      const holder = pdf.closest(".att-pdf");
      const existing = holder && holder.querySelector("iframe");
      if (existing) { existing.remove(); pdf.textContent = "Show here"; return; }
      const frame = document.createElement("iframe");
      frame.className = "pdf-frame";
      frame.src = pdf.dataset.src;
      frame.title = "PDF";
      holder.appendChild(frame);
      pdf.textContent = "Hide";
      return;
    }
    const older = target.closest(".load-older, .load-newer");
    if (older) { event.preventDefault(); loadThread(older.closest(".sentinel")); }
  });

  // ---------------------------------------------------------------- people autocomplete

  function setupPeople() {
    const input = document.querySelector(".people-input");
    const list = document.getElementById("people-list");
    if (!input || !list) return;
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const parts = input.value.split(",");
        const current = parts.pop().trim();
        const prefix = parts.map((p) => p.trim()).filter(Boolean);
        if (current.length < 1) return;
        try {
          const people = await fetchJSON("/api/people?q=" + encodeURIComponent(current));
          list.replaceChildren();
          for (const person of people) {
            const option = document.createElement("option");
            option.value = prefix.concat([person.short]).join(", ");
            option.label = person.name + " (" + person.count + ")";
            option.textContent = person.name;
            list.appendChild(option);
          }
        } catch (_error) { /* autocomplete is optional */ }
      }, 150);
    });
  }

  // ---------------------------------------------------------------- thread view

  function dedupeDayBreaks(container) {
    let previous = null;
    container.querySelectorAll(".day-break").forEach((el) => {
      if (previous && previous.dataset.day === el.dataset.day) {
        // A later break repeating the day that is already open adds nothing.
        const between = [];
        let node = previous.nextElementSibling;
        while (node && node !== el) { between.push(node); node = node.nextElementSibling; }
        if (between.every((n) => !n.classList.contains("day-break"))) el.remove();
        else previous = el;
      } else {
        previous = el;
      }
    });
  }

  async function loadThread(sentinel) {
    const thread = document.getElementById("thread");
    if (!sentinel || !thread || sentinel.dataset.loading || !sentinel.dataset.cursor) return;
    sentinel.dataset.loading = "1";
    const direction = sentinel.dataset.dir;
    const params = new URLSearchParams({ cursor: sentinel.dataset.cursor, dir: direction });
    if (thread.dataset.q) params.set("q", thread.dataset.q);
    try {
      const answer = await fetchJSON("/thread/" + encodeURIComponent(thread.dataset.thread) + "/messages?" + params.toString());
      const messages = thread.querySelector(".messages");
      const content = fragment(answer.html || "");
      if (direction === "older") {
        const before = document.documentElement.scrollHeight;
        messages.insertBefore(content, messages.firstChild);
        window.scrollBy(0, document.documentElement.scrollHeight - before);
      } else {
        messages.appendChild(content);
      }
      dedupeDayBreaks(messages);
      if (answer.more && answer.cursor) {
        sentinel.dataset.cursor = answer.cursor;
        delete sentinel.dataset.loading;
      } else {
        const edge = document.createElement("div");
        edge.className = "edge";
        edge.textContent = direction === "older" ? "Start of conversation" : "Latest message";
        sentinel.replaceWith(edge);
      }
    } catch (error) {
      delete sentinel.dataset.loading;
      sentinel.textContent = "Could not load: " + error.message;
    }
  }

  function setupThread() {
    const thread = document.getElementById("thread");
    if (!thread) return;
    const anchor = thread.querySelector(".msg.anchor") || (!window.location.hash && thread.querySelector(".msg:last-of-type"));
    if (anchor) anchor.scrollIntoView({ block: window.location.hash ? "center" : "end" });
    if (!("IntersectionObserver" in window)) return;
    // Created after the anchor is scrolled into view, so the first callback
    // sees the real position; prepending keeps it (see loadThread).
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) loadThread(entry.target);
      }
    }, { rootMargin: "600px 0px" });
    thread.querySelectorAll(".sentinel").forEach((el) => observer.observe(el));
  }

  // ---------------------------------------------------------------- keyboard

  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
    const active = document.activeElement;
    if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT")) return;
    const box = document.querySelector(".q");
    if (box) { event.preventDefault(); box.focus(); box.select(); }
  });

  document.addEventListener("DOMContentLoaded", async () => {
    const results = document.getElementById("results");
    if (results) {
      if ("scrollRestoration" in history) history.scrollRestoration = "manual";
      await restorePlace(results);
      let pending = null;
      window.addEventListener("scroll", () => {
        if (pending) return;
        pending = setTimeout(() => {
          pending = null;
          savePlace(Object.assign({ y: window.scrollY }, topOfScreen()));
        }, 200);
      }, { passive: true });
      watchSentinels(results);
      runSemantic();
    }
    setupPeople();
    setupThread();
  });
})();
