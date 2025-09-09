# app.py
# Improved Bioengineer Research Assistant
# - Bug fixes (clipboard, placeholders, duplicates)
# - Pagination for article results
# - Export workspace to ZIP (JSON + FASTA)
# - Better error handling & UX improvements

import os
import io
import json
import zipfile
import re
from collections import Counter
from typing import List, Dict, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from xml.etree import ElementTree as ET
import streamlit.components.v1 as components

st.set_page_config(page_title="Research Tool", layout="wide")
st.title("Research Tool")

# -------------------------
# Constants / endpoints
# -------------------------
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RCSB_PDB_URL = "https://files.rcsb.org/download/"

# -------------------------
# Session scaffolding (dedup keys)
# -------------------------
if "articles" not in st.session_state:
    st.session_state.articles = []         # list of dicts
    st.session_state._article_ids = set()
if "sequences" not in st.session_state:
    st.session_state.sequences = []
    st.session_state._sequence_ids = set()
if "molecules" not in st.session_state:
    st.session_state.molecules = []
    st.session_state._molecule_ids = set()
if "pdbs" not in st.session_state:
    st.session_state.pdbs = []
    st.session_state._pdb_ids = set()

# -------------------------
# Utilities
# -------------------------
def safe_findtext(elem: ET.Element, path: str) -> Optional[str]:
    t = elem.findtext(path)
    return t.strip() if isinstance(t, str) else None

def ncbi_request(path: str, params: dict, timeout: int = 20) -> requests.Response:
    api_key = st.session_state.get("ncbi_api_key") or os.environ.get("NCBI_API_KEY")
    p = params.copy()
    if api_key:
        p["api_key"] = api_key
    resp = requests.get(BASE_URL + path, params=p, timeout=timeout)
    resp.raise_for_status()
    return resp

# -----------------------
# AI Summarization Utils
# -----------------------
def get_api_key():
    """Fetch API key from env or Streamlit secrets"""
    return os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    
def chunk_text(text, max_chars=3000):
    """Split long texts into chunks for summarization"""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def analyze_with_ai(text: str):
    """
    Sends article text to the OpenRouter API and extracts summary, insights, and keywords.
    Always returns 3 values (summary, insights, keywords).
    """
    import os, requests, json
    api_key = get_api_key()
    if not api_key:
        return "❌ API key missing. Please set OPENROUTER_API_KEY.", "", []

    prompt = f"""
    Please analyze the following biomedical research text. 
    1. Provide a **concise summary** (3-5 sentences).  
    2. Extract **key insights or findings** useful for a bioengineer.  
    3. List the **most relevant keywords**.  

    Text:
    {text[:4000]}  # limit to avoid token overflow
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": prompt}],
            }),
            timeout=60,
        )

        if response.status_code != 200:
            return f"❌ API error: {response.text}", "", []

        content = response.json()["choices"][0]["message"]["content"]

        # crude parsing
        summary = content.split("Key Insights:")[0].strip()
        insights = ""
        keywords = []

        if "Key Insights:" in content:
            parts = content.split("Key Insights:")
            summary = parts[0].strip()
            insights_section = parts[1].split("Keywords:")[0].strip() if "Keywords:" in parts[1] else parts[1].strip()
            insights = insights_section
            if "Keywords:" in parts[1]:
                keywords = [k.strip() for k in parts[1].split("Keywords:")[1].split(",")]

        return summary, insights, keywords

    except Exception as e:
        return f"❌ Exception: {str(e)}", "", []


# -------------------------
# Caching wrappers
# -------------------------
@st.cache_data(show_spinner=False)
def search_ids(query: str, db: str = "pubmed", max_results: int = 10) -> List[str]:
    try:
        params = {"db": db, "term": query, "retmax": max_results, "retmode": "json"}
        r = ncbi_request("esearch.fcgi", params)
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        st.warning(f"search_ids error for {db}: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_pubmed(pmid: str, abstract_max_len: int = 800) -> Dict:
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = ncbi_request("efetch.fcgi", params)
        root = ET.fromstring(r.content)
        art = root.find(".//PubmedArticle")
        if art is None:
            return {"id": pmid, "error": "Not found"}
        title = safe_findtext(art, ".//ArticleTitle") or "No title"
        year = safe_findtext(art, ".//JournalIssue/PubDate/Year")
        if year is None:
            med = safe_findtext(art, ".//JournalIssue/PubDate/MedlineDate")
            if med:
                year = med.split()[0]
        journal = safe_findtext(art, ".//Journal/Title")
        abstract_parts = [t.text for t in art.findall(".//AbstractText") if t.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract"
        preview = abstract if len(abstract) <= abstract_max_len else abstract[:abstract_max_len] + "..."
        return {"db":"pubmed","id":pmid,"title":title,"year":year,"journal":journal,"abstract":abstract,"preview":preview}
    except Exception as e:
        return {"id": pmid, "error": str(e)}

@st.cache_data(show_spinner=False)
def fetch_pmc(pmcid: str, full_article: bool = False, abstract_max_len: int = 800) -> Dict:
    try:
        params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
        r = ncbi_request("efetch.fcgi", params)
        root = ET.fromstring(r.content)
        title = safe_findtext(root, ".//article-title") or safe_findtext(root, ".//title") or "No title"
        paragraphs = []
        for p in root.findall(".//body//p"):
            text_chunks = []
            if p.text:
                text_chunks.append(p.text)
            for node in p:
                if node.tail:
                    text_chunks.append(node.tail)
            paragraph = " ".join([t.strip() for t in text_chunks if t and t.strip()])
            if paragraph:
                paragraphs.append(paragraph)
        full_text = "\n\n".join(paragraphs).strip()
        if not full_text:
            sec_pars = [sec.text for sec in root.findall(".//sec//p") if sec.text]
            full_text = "\n\n".join(sec_pars)
        if full_article:
            return {"db":"pmc","id":pmcid,"title":title,"full_text":full_text or "Full text not available"}
        else:
            preview = (full_text[:abstract_max_len] + "...") if full_text and len(full_text) > abstract_max_len else (full_text or "No preview")
            return {"db":"pmc","id":pmcid,"title":title,"abstract":preview,"full_text":None}
    except Exception as e:
        return {"id": pmcid, "error": str(e)}

@st.cache_data(show_spinner=False)
def fetch_sequence(seq_id: str, db: str = "protein") -> Dict:
    try:
        params = {"db": db, "id": seq_id, "rettype": "fasta", "retmode": "text"}
        r = ncbi_request("efetch.fcgi", params)
        fasta = r.text.strip()
        return {"db": db, "id": seq_id, "fasta": fasta}
    except Exception as e:
        return {"db": db, "id": seq_id, "error": str(e)}

# -------------------------
# PubChem helpers
# -------------------------
@st.cache_data(show_spinner=False)
def pubchem_search_name_or_cid(query: str) -> Optional[int]:
    q = query.strip()
    try:
        if q.isdigit():
            return int(q)
        url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.requote_uri(q)}/cids/JSON"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if cids:
            return int(cids[0])
    except Exception as e:
        st.warning(f"PubChem search error: {e}")
    return None

@st.cache_data(show_spinner=False)
def pubchem_get_2d_png_bytes(cid: int) -> Optional[bytes]:
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/PNG"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.warning(f"PubChem PNG fetch error: {e}")
        return None

@st.cache_data(show_spinner=False)
def pubchem_get_3d_sdf(cid: int) -> Optional[str]:
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type=3d"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.warning(f"PubChem 3D SDF fetch error: {e}")
        return None

@st.cache_data(show_spinner=False)
def pubchem_get_properties(cid: int) -> Dict:
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        return props
    except Exception as e:
        st.warning(f"PubChem property fetch error: {e}")
        return {}

# -------------------------
# PDB helpers
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_pdb_file(pdb_id: str) -> Optional[str]:
    pdb_id = pdb_id.strip().lower()
    try:
        url = RCSB_PDB_URL + f"{pdb_id}.pdb"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.warning(f"Failed to fetch PDB {pdb_id}: {e}")
        return None

def extract_pdb_ligands(pdb_text: str) -> List[str]:
    ligands = set()
    for line in pdb_text.splitlines():
        if line.startswith("HET   "):
            # crude parse: residue name typically at cols 7-10, but easiest via split
            parts = line.split()
            if len(parts) >= 4:
                lig_name = parts[3]
                if lig_name not in ("HOH", "WAT"):
                    ligands.add(lig_name)
        elif line.startswith("HETATM"):
            parts = line.split()
            if len(parts) >= 4:
                lig_name = parts[3]
                if lig_name not in ("HOH", "WAT"):
                    ligands.add(lig_name)
    return sorted(ligands)

# -------------------------
# Visualizations
# -------------------------
def plot_aa_composition(fasta: str):
    seq = "".join([line.strip() for line in fasta.splitlines() if not line.startswith(">")])
    counts = Counter(seq)
    if not counts:
        st.info("No sequence to plot")
        return
    aa = sorted(counts.keys())
    freq = [counts[a] for a in aa]
    fig, ax = plt.subplots(figsize=(9,3))
    ax.bar(aa, freq)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def kyte_doolittle(seq: str, window: int = 9):
    kd = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
        "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
        "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
        "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
    }
    seq = seq.upper()
    scores = []
    half = window // 2
    for i in range(len(seq)):
        start = max(0, i - half)
        end = min(len(seq), i + half + 1)
        window_seq = seq[start:end]
        vals = [kd.get(aa, 0) for aa in window_seq]
        scores.append(sum(vals) / len(vals) if vals else 0)
    return scores

def plot_hydrophobicity(fasta: str, window: int = 9):
    seq = "".join([line.strip() for line in fasta.splitlines() if not line.startswith(">")])
    if not seq:
        st.info("No sequence to plot hydrophobicity")
        return
    scores = kyte_doolittle(seq, window=window)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(range(1, len(scores)+1), scores)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Kyte-Doolittle hydrophobicity")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    st.pyplot(fig)

def plot_gc_and_length(fasta: str):
    seq = "".join([line.strip() for line in fasta.splitlines() if not line.startswith(">")]).upper()
    if not seq:
        st.info("No sequence")
        return
    length = len(seq)
    gc = (seq.count("G") + seq.count("C")) / length * 100 if length > 0 else 0
    col1, col2 = st.columns(2)
    col1.metric("GC content (%)", f"{gc:.2f}")
    col2.metric("Sequence length", f"{length:,}")

# -------------------------
# 3D HTML builders (3Dmol.js)
# -------------------------
def make_3dmol_html_from_sdf(sdf_text: str, width: int = 700, height: int = 450):
    import json
    sdf_js = json.dumps(sdf_text)
    html = f"""
<div id="viewer" style="width:{width}px; height:{height}px; position: relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  const sdf = {sdf_js};
  viewer.addModel(sdf, "sdf");
  viewer.setStyle({{}}, {{stick:{{}}}});
  viewer.zoomTo();
  viewer.render();
</script>
"""
    return html

def make_3dmol_html_from_pdb(pdb_text: str, width: int = 800, height: int = 500):
    import json
    pdb_js = json.dumps(pdb_text)
    html = f"""
<div id="viewer" style="width:{width}px; height:{height}px; position: relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  const pdb = {pdb_js};
  viewer.addModel(pdb, "pdb");
  viewer.setStyle({{}}, {{cartoon:{{color: 'spectrum'}}}});
  viewer.zoomTo();
  viewer.render();
</script>
"""
    return html

# -------------------------
# Highlight helper
# -------------------------
def highlight_terms(text: str, terms: List[str]) -> str:
    if not text:
        return ""
    out = text
    terms_sorted = sorted(set([t for t in terms if t.strip()]), key=lambda x: -len(x))
    for t in terms_sorted:
        pattern = re.compile(re.escape(t), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    out = out.replace("\n", "<br>")
    return out

# -------------------------
# Simple co-occurrence network
# -------------------------
def build_cooccurrence_graph(docs: List[str], top_n_terms: int = 30):
    words = []
    for d in docs:
        if not d: continue
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", d.lower())
        words.extend(tokens)
    common = [w for w, _ in Counter(words).most_common(top_n_terms)]
    G = nx.Graph()
    for w in common:
        G.add_node(w, size=Counter(words)[w])
    for d in docs:
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", (d or "").lower()))
        tokens = tokens.intersection(set(common))
        for a in tokens:
            for b in tokens:
                if a != b:
                    G.add_edge(a, b, weight=G.get_edge_data(a, b, {}).get("weight", 0) + 1)
    return G

def plot_network(G):
    if G.number_of_nodes() == 0:
        st.info("No network to show")
        return
    pos = nx.spring_layout(G, seed=1)
    sizes = [300 + 20 * G.nodes[n].get("size", 1) for n in G.nodes()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    fig, ax = plt.subplots(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[max(0.3, w/2) for w in weights], alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.axis('off')
    st.pyplot(fig)

# -------------------------
# Clipboard helper (JS)
# -------------------------
def js_copy_button(text: str, key: str = "copy"):
    # returns a component with a copy button that copies `text` to clipboard
    escaped = json.dumps(text)  # safe JS string
    html = f"""
<button id="btn_{key}">Copy</button>
<script>
const txt = {escaped};
document.getElementById("btn_{key}").addEventListener("click", async () => {{
  await navigator.clipboard.writeText(txt);
  const prev = document.getElementById("btn_{key}");
  prev.innerText = "Copied!";
  setTimeout(()=> prev.innerText="Copy", 1500);
}});
</script>
"""
    components.html(html, height=40)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Controls")
st.sidebar.text_input("NCBI API key (optional)", key="ncbi_api_key")

section = st.sidebar.radio("Section", ["Articles", "Sequences", "Molecules", "Protein 3D", "Workspace", "Network"])

# -------------------------
# Articles tab (revamped with AI features)
# -------------------------
if section == "Articles":
    st.sidebar.header("Article search")
    query = st.sidebar.text_input("Query", value="pancreas organ on a chip")
    max_results = st.sidebar.number_input("Max results per DB", min_value=1, max_value=20, value=6, step=1)
    full_article_toggle = st.sidebar.checkbox("Fetch full text from PMC (if available)", value=False)
    year_from, year_to = st.sidebar.slider("Year range", 1990, 2026, (2000, 2025))
    organ_terms = st.sidebar.text_input("Highlight keywords (comma-separated)", value="pancreas, insulin, islet")
    highlight_list = [t.strip() for t in organ_terms.split(",") if t.strip()]

    # pagination state
    if "articles_page" not in st.session_state:
        st.session_state.articles_page = 0
    per_page = st.sidebar.number_input("Results per page (UI)", min_value=1, max_value=10, value=3)

    if st.sidebar.button("Search"):
        st.session_state.articles_page = 0
        with st.spinner("🔎 Searching PubMed and PMC..."):
            pmids = search_ids(query, db="pubmed", max_results=max_results)
            pmcs = search_ids(query, db="pmc", max_results=max_results)
            results_display = []

            # PubMed
            for pmid in pmids:
                res = fetch_pubmed(pmid, abstract_max_len=2000)
                try:
                    y = int(res.get("year")) if res.get("year") else None
                except:
                    y = None
                if y is not None and not (year_from <= y <= year_to):
                    continue
                res["db"] = "pubmed"
                results_display.append(res)

            # PMC
            for pid in pmcs:
                res = fetch_pmc(pid, full_article=full_article_toggle, abstract_max_len=2000)
                res["db"] = "pmc"
                results_display.append(res)

            st.session_state._last_search_results = results_display

    # show results
    results_display = st.session_state.get("_last_search_results", [])
    total = len(results_display)

    if total == 0:
        st.info("ℹ️ No results yet — run a search.")
    else:
        page = st.session_state.articles_page
        start = page * per_page
        end = min(total, start + per_page)
        st.markdown(f"**📄 Showing results {start+1} — {end} of {total}**")

        for r in results_display[start:end]:
            title = r.get("title") or r.get("id")
            year = r.get("year") or ""
            journal = r.get("journal") or ""
            db = r.get("db")

            with st.expander(f"**{title}** ({year})", expanded=False):
                if journal:
                    st.caption(f"📚 *{journal}*")

                # Links
                if db == "pmc":
                    pid = r.get("id")
                    if str(pid).lower().startswith("pmc"):
                        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pid}"
                    else:
                        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}"
                    st.markdown(f"[🔗 Open in PMC]({url})")
                elif db == "pubmed":
                    st.markdown(f"[🔗 Open in PubMed](https://pubmed.ncbi.nlm.nih.gov/{r.get('id')})")

                # Show abstract/full text
                text_to_show = r.get("full_text") or r.get("abstract") or r.get("preview")
                if text_to_show:
                    html = highlight_terms(text_to_show, highlight_list)
                    st.markdown("**Article content:**")
                    st.markdown(html, unsafe_allow_html=True)
                    st.download_button("⬇️ Download text", data=text_to_show, file_name=f"{r.get('id')}.txt")
                else:
                    st.warning("No text available.")

                # --- AI Analysis ---
                if st.button("🤖 Summarize & Extract Insights", key=f"summarize_{db}_{r.get('id')}"):
                    with st.spinner("AI is analyzing the text..."):
                        summary, insights, keywords = analyze_with_ai(text_to_show)
                        st.subheader("🔑 Summary")
                        st.write(summary)
                        st.subheader("💡 Key Insights")
                        st.write(insights)
                        st.subheader("🧬 Keywords")
                        st.write(", ".join(keywords))

                # Actions row
                cols = st.columns([1,1,1,3])
                if cols[0].button("Save", key=f"save_article_{db}_{r.get('id')}"):
                    uid = f"{db}:{r.get('id')}"
                    if uid not in st.session_state._article_ids:
                        st.session_state.articles.append(r)
                        st.session_state._article_ids.add(uid)
                        st.success("✅ Saved")
                    else:
                        st.info("Already saved")

                if cols[1].button("Export JSON", key=f"export_article_{db}_{r.get('id')}"):
                    st.download_button("Download JSON", data=json.dumps(r, indent=2).encode("utf-8"), file_name=f"{r.get('id')}.json", mime="application/json")

                if cols[2].button("Copy citation", key=f"cite_article_{db}_{r.get('id')}"):
                    citation = f"{r.get('title')} ({r.get('year')}) {r.get('journal')}"
                    js_copy_button(citation, key=f"cite_{r.get('id')}")

        # Pagination
        page_cols = st.columns([1,1,6])
        if page_cols[0].button("⬅️ Previous") and st.session_state.articles_page > 0:
            st.session_state.articles_page -= 1
        if page_cols[1].button("Next ➡️") and (st.session_state.articles_page+1)*per_page < total:
            st.session_state.articles_page += 1

# -------------------------
# Sequences tab
# -------------------------
elif section == "Sequences":
    st.sidebar.header("Sequence fetch & analysis")
    seq_id_input = st.sidebar.text_input("Sequence ID(s) (comma-separated)", value="")
    seq_db = st.sidebar.selectbox("Database", ["protein","nucleotide"])
    motif = st.sidebar.text_input("Search motif (regex, optional)", value="")
    hyd_window = st.sidebar.slider("Hydrophobicity window", 5, 21, 9, step=2)
    if st.sidebar.button("Fetch sequences"):
        ids = [i.strip() for i in seq_id_input.split(",") if i.strip()]
        if not ids:
            st.sidebar.error("Provide at least one ID")
        for sid in ids:
            rec = fetch_sequence(sid, db=seq_db)
            if rec.get("error"):
                st.error(f"Error fetching {sid}: {rec['error']}")
                continue
            uid = f"{rec.get('db')}:{rec.get('id')}"
            if uid not in st.session_state._sequence_ids:
                st.session_state.sequences.append(rec)
                st.session_state._sequence_ids.add(uid)
            # Display
            st.subheader(f"{rec.get('db').upper()} — {rec.get('id')}")
            fasta_preview = "\n".join(rec.get("fasta","").splitlines()[:200])
            st.code(fasta_preview)
            if seq_db == "protein":
                plot_aa_composition(rec.get("fasta",""))
                st.markdown("Hydrophobicity (Kyte–Doolittle)")
                plot_hydrophobicity(rec.get("fasta",""), window=hyd_window)
            else:
                plot_gc_and_length(rec.get("fasta",""))
            if motif:
                seq = "".join([l for l in rec.get("fasta","").splitlines() if not l.startswith(">")])
                matches = [m.span() for m in re.finditer(motif, seq)]
                st.write(f"Motif matches (start,end): {matches if matches else 'None'}")

# -------------------------
# Molecules tab
# -------------------------
elif section == "Molecules":
    st.sidebar.header("PubChem lookup")
    mol_query = st.sidebar.text_input("Name or CID", value="glucose")
    if st.sidebar.button("Lookup molecule"):
        with st.spinner("Searching PubChem..."):
            cid = pubchem_search_name_or_cid(mol_query)
            if not cid:
                st.error("No PubChem CID found")
            else:
                props = pubchem_get_properties(cid)
                png = pubchem_get_2d_png_bytes(cid)
                sdf = pubchem_get_3d_sdf(cid)
                st.subheader(f"PubChem CID {cid}")
                if props:
                    st.write("**Properties**")
                    st.json(props)
                if png:
                    st.image(png, caption=f"{mol_query} (CID {cid})")
                if sdf:
                    st.markdown("**3D interactive view (3Dmol.js)**")
                    html = make_3dmol_html_from_sdf(sdf)
                    components.html(html, height=520, scrolling=False)
                link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
                st.markdown(f"[Open in PubChem]({link})")
                uid = f"pubchem:{cid}"
                if uid not in st.session_state._molecule_ids:
                    st.session_state.molecules.append({"query":mol_query,"cid":cid,"props":props})
                    st.session_state._molecule_ids.add(uid)

# -------------------------
# Protein 3D tab
# -------------------------
elif section == "Protein 3D":
    st.sidebar.header("RCSB PDB viewer")
    pdb_id_input = st.sidebar.text_input("PDB ID (e.g., 1TUP)", value="")
    if st.sidebar.button("Load PDB"):
        if not pdb_id_input:
            st.error("Provide a PDB ID")
        else:
            with st.spinner("Fetching PDB..."):
                pdb_text = fetch_pdb_file(pdb_id_input)
                if not pdb_text:
                    st.error("Could not fetch PDB")
                else:
                    uid = f"pdb:{pdb_id_input.lower()}"
                    if uid not in st.session_state._pdb_ids:
                        st.session_state.pdbs.append({"pdb_id":pdb_id_input})
                        st.session_state._pdb_ids.add(uid)
                    st.subheader(f"PDB: {pdb_id_input}")
                    ligs = extract_pdb_ligands(pdb_text)
                    if ligs:
                        st.markdown("**Ligands (non-water)**")
                        st.write(ligs)
                    html = make_3dmol_html_from_pdb(pdb_text)
                    components.html(html, height=560, scrolling=False)

# -------------------------
# Workspace tab (export to zip)
# -------------------------
elif section == "Workspace":
    st.header("Session workspace")
    st.markdown("Saved items during this session")
    # Articles
    st.subheader("Articles")
    if st.session_state.articles:
        for a in st.session_state.articles:
            with st.expander(a.get("title") or a.get("id")):
                st.write({"id": a.get("id"), "title": a.get("title"), "year": a.get("year"), "journal": a.get("journal")})
    else:
        st.info("No articles saved")
    # Sequences
    st.subheader("Sequences")
    if st.session_state.sequences:
        table = [{"db": s.get("db"), "id": s.get("id"), "length": len("".join([l for l in s.get("fasta","").splitlines() if not l.startswith(">")]))} for s in st.session_state.sequences]
        st.table(pd.DataFrame(table))
    else:
        st.info("No sequences saved")
    # Molecules
    st.subheader("Molecules")
    if st.session_state.molecules:
        st.table(pd.DataFrame(st.session_state.molecules))
    else:
        st.info("No molecules saved")
    # PDBs
    st.subheader("PDBs")
    if st.session_state.pdbs:
        st.table(pd.DataFrame(st.session_state.pdbs))
    else:
        st.info("No pdbs saved")

    st.markdown("---")
    # Export to ZIP
    if st.button("Export workspace as ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("articles.json", json.dumps(st.session_state.articles, indent=2))
            # sequences as FASTA files
            for s in st.session_state.sequences:
                fasta = s.get("fasta","")
                if fasta:
                    fname = f"{s.get('db')}_{s.get('id')}.fasta"
                    z.writestr(fname, fasta)
            z.writestr("molecules.json", json.dumps(st.session_state.molecules, indent=2))
            z.writestr("pdbs.json", json.dumps(st.session_state.pdbs, indent=2))
        buf.seek(0)
        st.download_button("Download workspace ZIP", data=buf.read(), file_name="workspace_export.zip", mime="application/zip")

    if st.button("Clear workspace"):
        st.session_state.articles.clear()
        st.session_state.sequences.clear()
        st.session_state.molecules.clear()
        st.session_state.pdbs.clear()
        st.session_state._article_ids.clear()
        st.session_state._sequence_ids.clear()
        st.session_state._molecule_ids.clear()
        st.session_state._pdb_ids.clear()
        st.success("Workspace cleared")

# -------------------------
# Network tab
# -------------------------
elif section == "Network":
    st.header("Co-occurrence network from saved article texts")
    docs = [a.get("abstract") or a.get("full_text") or "" for a in st.session_state.articles]
    docs = [d for d in docs if d]
    if not docs:
        st.info("No article texts saved. Save articles first.")
    else:
        top_n = st.slider("Top terms to include", 10, 100, 40)
        G = build_cooccurrence_graph(docs, top_n_terms=top_n)
        plot_network(G)
        if st.button("Export network JSON"):
            data = {"nodes":[{"id":n,"size":G.nodes[n].get("size",1)} for n in G.nodes()],
                    "links":[{"source":u,"target":v,"weight":G[u][v]["weight"]} for u,v in G.edges()]}
            st.download_button("Download network", data=json.dumps(data, indent=2).encode("utf-8"), file_name="network.json", mime="application/json")

# -------------------------
# Footer tips
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Tips: use the Workspace to collect items, export as ZIP for sharing. If you plan heavy use, set an NCBI API key in the sidebar (or the NCBI_API_KEY env var).")
