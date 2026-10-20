// Browser client for the REST API. Uses the session cookie and Django's CSRF token.
const $ = (id) => document.getElementById(id);
let current = null;

function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

async function api(path, { method = "GET", json, form } = {}) {
  const headers = { "X-CSRFToken": csrfToken(), Accept: "application/json" };
  let body;
  if (json) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  const response = await fetch(`/api${path}`, { method, headers, body, credentials: "same-origin" });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    const detail = data?.detail ?? Object.values(data ?? {}).flat().join(" ");
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}

function item(text, { onClick, active = false, extra = "" } = {}) {
  const li = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = `link${active ? " active" : ""}`;
  button.textContent = text;
  if (onClick) button.addEventListener("click", onClick);
  li.append(button);
  if (extra) {
    const span = document.createElement("span");
    span.className = "muted small";
    span.textContent = ` ${extra}`;
    li.append(span);
  }
  return li;
}

function setStatus(text) {
  $("status").textContent = text;
}

async function loadCollections(selectId) {
  const { results } = await api("/collections/");
  const list = $("collections");
  list.replaceChildren(
    ...results.map((c) =>
      item(c.name, { active: c.id === selectId, extra: `${c.document_count} docs`, onClick: () => select(c) }),
    ),
  );
  const chosen = results.find((c) => c.id === selectId) ?? results[0];
  if (chosen && chosen.id !== current?.id) select(chosen);
  if (!results.length) $("collection-title").textContent = "· create a collection first";
}

async function select(collection) {
  current = collection;
  $("collection-title").textContent = `· ${collection.name}`;
  for (const button of $("collections").querySelectorAll("button")) {
    button.classList.toggle("active", button.textContent === collection.name);
  }
  $("answer").hidden = true;
  await loadDocuments();
}

async function loadDocuments() {
  const { results } = await api(`/collections/${current.id}/documents/`);
  $("documents").replaceChildren(
    ...(results.length
      ? results.map((d) => item(d.title, { extra: `${d.chunk_count} chunks · ${d.status}` }))
      : [Object.assign(document.createElement("li"), { className: "muted", textContent: "No documents yet." })]),
  );
}

function showAnswer(data) {
  $("answer-text").textContent = data.answer;
  const usage = data.model ? ` · ${data.model} · ${data.input_tokens} in / ${data.output_tokens} out tokens` : "";
  $("answer-meta").textContent = `${data.grounded ? "Grounded in your documents" : "Not grounded"} · ${data.provider}${usage} · ${data.latency_ms} ms`;
  const cited = new Map(data.citations.map((c) => [c.chunk_id, c.cited_text]));
  $("sources").replaceChildren(
    ...data.sources.map((s) => {
      const li = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = `${s.document_title} `;
      const score = document.createElement("span");
      score.className = "muted small";
      score.textContent = `similarity ${s.similarity.toFixed(2)}`;
      const quote = document.createElement("blockquote");
      quote.textContent = cited.get(s.chunk_id) ?? s.text;
      if (cited.has(s.chunk_id)) li.className = "cited";
      li.append(title, score, quote);
      return li;
    }),
  );
  $("answer").hidden = false;
}

$("new-collection").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const created = await api("/collections/", { method: "POST", json: { name: $("collection-name").value } });
    event.target.reset();
    await loadCollections(created.id);
  } catch (error) {
    setStatus(error.message);
  }
});

$("upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!current) return setStatus("Create a collection first.");
  const form = new FormData(event.target);
  setStatus("Uploading and indexing…");
  try {
    const doc = await api(`/collections/${current.id}/documents/`, { method: "POST", form });
    setStatus(`Indexed “${doc.title}” into ${doc.chunk_count} chunks.`);
    event.target.reset();
    await loadCollections(current.id);
    await loadDocuments();
  } catch (error) {
    setStatus(error.message);
  }
});

$("ask").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!current) return setStatus("Create a collection first.");
  setStatus("Searching and answering…");
  try {
    showAnswer(await api(`/collections/${current.id}/ask/`, { method: "POST", json: { question: $("question").value } }));
    setStatus("");
  } catch (error) {
    setStatus(error.message);
  }
});

loadCollections().catch((error) => setStatus(error.message));
