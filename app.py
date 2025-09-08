# app.py
import os
import io
import json
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from xml.etree import ElementTree as ET
import streamlit.components.v1 as components

# -------------------------
# Basic config
# -------------------------
st.set_page_config(page_title="Bioengineer Research Assistant", layout="wide")
st.title("Researsch tool")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RCSB_PDB_URL = "https://files.rcsb.org/download/"

# -------------------------
# Session workspace
# -------------------------
if "articles" not in st.session_state:
    st.session_state.articles = []  # list of dicts
if "sequences" not in st.session_state:
    st.session_state.sequences = []
if "molecules" not in st.session_state:
    st.session_state.molecules = []
if "pdbs" not in st.session_state:
    st.session_state.pdbs = []

# -------------------------
# Utility helpers
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

# -------------------------
# CACHED API calls
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
def fetch_pubmed(pmid: str, abstract_max_len: int = 500) -> Dict:
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = ncbi_request("efetch.fcgi", params)
        root = ET.fromstring(r.content)
        art = root.find(".//PubmedArticle")
        if art is None:
            return {"id": pmid, "error": "Not found"}
        title = safe_findtext(art, ".//ArticleTitle") or "No title"
        year = safe_findtext(art, ".//JournalIssue/PubDate/Year")
        # sometimes MedlineDate exists
        if year is None:
            med = safe_findtext(art, ".//JournalIssue/PubDate/MedlineDate")
            if med:
                year = med.split()[0]
        journal = safe_findtext(art, ".//Journal/Title")
        abstract_parts = [t.text for t in art.findall(".//AbstractText") if t.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract"
        if len(abstract) > abstract_max_len:
            preview = abstract[:abstract_max_len] + "..."
        else:
            preview = abstract
        return {"db":"pubmed","id":pmid,"title":title,"year":year,"journal":journal,"abstract":abstract,"preview":preview}
    except Exception as e:
        return {"id": pmid, "error": str(e)}

@st.cache_data(show_spinner=False)
def fetch_pmc(pmcid: str, full_article: bool = False, abstract_max_len: int = 500) -> Dict:
    try:
        params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
        r = ncbi_request("efetch.fcgi", params)
        root = ET.fromstring(r.content)
        title = safe_findtext(root, ".//article-title") or safe_findtext(root, ".//title") or "No title"
        paragraphs = []
        # gather paragraphs in body
        for p in root.findall(".//body//p"):
            # combine text + tails
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
            # fallback
            sec_pars = [sec.text for sec in root.findall(".//sec//p") if sec.text]
            full_text = "\n\n".join(sec_pars)
        if full_article:
            return {"db":"pmc","id":pmcid,"title":title,"full_text":full_text or "Full text not available"}
        else:
            preview = (full_text[:abstract_max_len] + "...") if full_text and len(full_text) > abstract_max_len else (full_text or "No preview available")
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
        # request common properties
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
# PDB fetcher and ligand parser
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
        if line.startswith("HET   ") or line.startswith("HETATM"):
            # HET lines: format uses residue name at columns 17-20 historically
            parts = line.split()
            # crude attempt: residue name likely at position 3 or 4
            if len(parts) >= 4:
                lig_name = parts[3]
                # filter common solvent names
                if lig_name not in ("HOH", "WAT"):
                    ligands.add(lig_name)
    return sorted(ligands)

# -------------------------
# Visualization helpers
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
    # Kyte-Doolittle hydrophobicity scale
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
# 3D viewer HTML builders
# -------------------------
def make_3dmol_html_from_sdf(sdf_text: str, width: int = 700, height: int = 450):
    import json
    sdf_js = json.dumps(sdf_text)
    # double braces for .format
    html = """
<div id="viewer" style="width:{w}px; height:{h}px; position: relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  const sdf = {sdf};
  viewer.addModel(sdf, "sdf");
  viewer.setStyle({{}}, {{stick:{{}}}});
  viewer.zoomTo();
  viewer.render();
</script>
""".format(w=width, h=height, sdf=sdf_js)
    return html

def make_3dmol_html_from_pdb(pdb_text: str, width: int = 800, height: int = 500):
    import json
    pdb_js = json.dumps(pdb_text)
    html = """
<div id="viewer" style="width:{w}px; height:{h}px; position: relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  const pdb = {pdb};
  viewer.addModel(pdb, "pdb");
  viewer.setStyle({{}}, {{cartoon:{{color: 'spectrum'}}}});
  viewer.zoomTo();
  viewer.render();
</script>
""".format(w=width, h=height, pdb=pdb_js)
    return html

# -------------------------
# Text highlighting utility
# -------------------------
def highlight_terms(text: str, terms: List[str]) -> str:
    if not text:
        return ""
    out = text
    # sort terms by length to avoid partial overlapping replacements
    terms_sorted = sorted(set([t for t in terms if t.strip()]), key=lambda x: -len(x))
    for t in terms_sorted:
        # case-insensitive replace with <mark>
        pattern = re.compile(re.escape(t), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    # escape newlines to <br>
    out = out.replace("\n", "<br>")
    return out

# -------------------------
# Simple co-occurrence network from abstracts
# -------------------------
def build_cooccurrence_graph(docs: List[str], top_n_terms: int = 30):
    # naive term extraction: split by non-alphanum and lowercase, count terms
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
# Sidebar controls: top-level organization
# -------------------------
st.sidebar.title("Controls")
st.sidebar.text_input("NCBI API key (optional)", key="ncbi_api_key")

section = st.sidebar.radio("Section", ["Articles", "Sequences", "Molecules", "Protein 3D", "Workspace", "Network"])

# -------------------------
# ARTICLES tab
# -------------------------
if section == "Articles":
    st.sidebar.header("Article search")
    query = st.sidebar.text_input("Query (PubMed/PMC)", value="pancreas organ on a chip")
    max_results = st.sidebar.slider("Max results per DB", 1, 20, 5)
    full_article_toggle = st.sidebar.checkbox("Fetch full text from PMC (if available)", value=False)
    year_from, year_to = st.sidebar.slider("Year range", 1990, 2026, (2000, 2025))
    organ_terms = st.sidebar.text_input("Highlight keywords (comma-separated)", value="pancreas, insulin, islet")
    highlight_list = [t.strip() for t in organ_terms.split(",") if t.strip()]
    if st.sidebar.button("Search"):
        with st.spinner("Searching PubMed and PMC..."):
            pmids = search_ids(query, db="pubmed", max_results=max_results)
            pmcs = search_ids(query, db="pmc", max_results=max_results)
            results_display = []
            # fetch pubmed results
            for pmid in pmids:
                res = fetch_pubmed(pmid, abstract_max_len=1500)
                # filter by year range if possible
                try:
                    y = int(res.get("year")) if res.get("year") else None
                except:
                    y = None
                if y is not None and not (year_from <= y <= year_to):
                    continue
                results_display.append(res)
                st.session_state.articles.append(res)
            # fetch pmc results
            for pid in pmcs:
                res = fetch_pmc(pid, full_article=full_article_toggle, abstract_max_len=1500)
                results_display.append(res)
                st.session_state.articles.append(res)
        # show literature dashboard
        st.subheader("Literature dashboard")
        years = [int(r.get("year")) for r in st.session_state.articles if r.get("year") and r.get("year").isdigit()]
        if years:
            df_year = pd.Series(years).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8,3))
            df_year.plot(kind="bar", ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        # top journals
        journals = [r.get("journal") for r in st.session_state.articles if r.get("journal")]
        if journals:
            topj = pd.Series(journals).value_counts().head(10)
            st.markdown("**Top journals (session)**")
            st.table(topj.reset_index().rename(columns={"index":"Journal", 0:"Count"}))
        # display results nicely
        st.subheader("Search results")
        for r in results_display:
            title = r.get("title") or r.get("id")
            year = r.get("year") or ""
            journal = r.get("journal") or ""
            with st.expander(f"{title}  {f'({year})' if year else ''}", expanded=False):
                st.markdown(f"**Journal:** {journal}")
                # links
                if r.get("db") == "pmc" or str(r.get("id")).lower().startswith("pmc"):
                    pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{r.get('id')}" if not str(r.get("id")).lower().startswith("pmc") else f"https://www.ncbi.nlm.nih.gov/pmc/articles/{r.get('id')}"
                    st.markdown(f"[PMC link]({pmc_link})")
                if r.get("db") == "pubmed" or r.get("id"):
                    st.markdown(f"[PubMed link](https://pubmed.ncbi.nlm.nih.gov/{r.get('id')})")
                # show abstract or full text
                if r.get("full_text"):
                    html = highlight_terms(r.get("full_text"), highlight_list)
                    st.markdown("**Full text (PMC)**")
                    st.markdown(html, unsafe_allow_html=True)
                    st.download_button("Download full text", data=r.get("full_text",""), file_name=f"{r.get('id')}_fulltext.txt")
                elif r.get("abstract"):
                    html = highlight_terms(r.get("abstract"), highlight_list)
                    st.markdown("**Abstract / preview**")
                    st.markdown(html, unsafe_allow_html=True)
                elif r.get("preview"):
                    html = highlight_terms(r.get("preview"), highlight_list)
                    st.markdown("**Preview**")
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("No text available")
                # tag or save
                cols = st.columns([1,1,1,3])
                if cols[0].button("Save to workspace", key=f"save_{r.get('id')}"):
                    st.session_state.articles.append(r)
                    st.success("Saved to workspace")
                if cols[1].button("Export as JSON", key=f"export_{r.get('id')}"):
                    st.download_button("Download JSON", data=json.dumps(r, indent=2).encode("utf-8"), file_name=f"{r.get('id')}.json", mime="application/json")
                if cols[2].button("Copy citation (simple)", key=f"cite_{r.get('id')}"):
                    citation = f"{r.get('title')} ({r.get('year')}) {r.get('journal')}"
                    st.clipboard_set(citation)
                    st.info("Citation copied to clipboard")

# -------------------------
# SEQUENCES tab
# -------------------------
elif section == "Sequences":
    st.sidebar.header("Sequence fetch & analysis")
    seq_id_input = st.sidebar.text_input("Sequence ID(s) (comma-separated, e.g., NP_000000 or accession)", value="")
    seq_db = st.sidebar.selectbox("Database", ["protein","nucleotide"])
    motif = st.sidebar.text_input("Search motif (optional, regex)")
    hyd_window = st.sidebar.slider("Hydrophobicity window", 5, 21, 9, step=2)
    if st.sidebar.button("Fetch sequences"):
        ids = [i.strip() for i in seq_id_input.split(",") if i.strip()]
        for sid in ids:
            rec = fetch_sequence(sid, db=seq_db)
            if rec.get("error"):
                st.error(f"Error fetching {sid}: {rec['error']}")
                continue
            st.session_state.sequences.append(rec)
            st.subheader(f"Sequence: {sid} ({seq_db})")
            st.code(rec.get("fasta","")[:2000])
            if seq_db == "protein":
                plot_aa_composition = plot_aa_composition if False else None  # placeholder to avoid lint
                plot_aa_composition(fasta=rec.get("fasta","")) if True else None
                st.markdown("Hydrophobicity (Kyte-Doolittle)")
                plot_hydrophobicity(rec.get("fasta",""), window=hyd_window)
                if motif:
                    seq = "".join([l for l in rec.get("fasta","").splitlines() if not l.startswith(">")])
                    matches = [m.span() for m in re.finditer(motif, seq)]
                    st.write(f"Motif matches (start,end): {matches}")
            else:
                plot_gc_and_length(rec.get("fasta",""))
                if motif:
                    seq = "".join([l for l in rec.get("fasta","").splitlines() if not l.startswith(">")])
                    matches = [m.span() for m in re.finditer(motif, seq)]
                    st.write(f"Motif matches (start,end): {matches}")

# -------------------------
# MOLECULES tab
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
                st.success(f"Found CID {cid}")
                props = pubchem_get_properties(cid)
                png = pubchem_get_2d_png_bytes(cid)
                sdf = pubchem_get_3d_sdf(cid)
                st.subheader(f"PubChem CID {cid}")
                if props:
                    st.write("**Properties**")
                    st.write(props)
                if png:
                    st.image(png, caption=f"{mol_query} (CID {cid})")
                if sdf:
                    st.markdown("**3D interactive view**")
                    html = make_3dmol_html_from_sdf(sdf)
                    components.html(html, height=500, scrolling=False)
                st.session_state.molecules.append({"query":mol_query,"cid":cid,"props":props})

# -------------------------
# PROTEIN 3D tab
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
                    st.session_state.pdbs.append({"pdb_id":pdb_id_input})
                    st.subheader(f"PDB: {pdb_id_input}")
                    ligs = extract_pdb_ligands(pdb_text)
                    if ligs:
                        st.markdown("**Ligands (non-water)**")
                        st.write(ligs)
                    html = make_3dmol_html_from_pdb(pdb_text)
                    components.html(html, height=560, scrolling=False)

# -------------------------
# WORKSPACE tab
# -------------------------
elif section == "Workspace":
    st.header("Session workspace")
    st.markdown("Saved items (articles, sequences, molecules, pdbs) from this session")
    st.subheader("Articles")
    if st.session_state.articles:
        for a in st.session_state.articles:
            with st.expander(f"{a.get('title') or a.get('id')}"):
                st.write(a)
    else:
        st.info("No articles saved in session")
    st.subheader("Sequences")
    if st.session_state.sequences:
        for s in st.session_state.sequences:
            st.code(s.get("fasta","")[:1000])
    else:
        st.info("No sequences in session")
    st.subheader("Molecules")
    if st.session_state.molecules:
        st.table(pd.DataFrame(st.session_state.molecules))
    else:
        st.info("No molecules saved")
    st.subheader("PDBs")
    if st.session_state.pdbs:
        st.table(pd.DataFrame(st.session_state.pdbs))
    else:
        st.info("No pdbs saved")
    st.markdown("---")
    if st.button("Export entire session JSON"):
        data = {
            "articles": st.session_state.articles,
            "sequences": st.session_state.sequences,
            "molecules": st.session_state.molecules,
            "pdbs": st.session_state.pdbs
        }
        blob = json.dumps(data, indent=2).encode("utf-8")
        st.download_button("Download session JSON", data=blob, file_name="session_export.json", mime="application/json")
    if st.button("Clear workspace"):
        st.session_state.articles.clear()
        st.session_state.sequences.clear()
        st.session_state.molecules.clear()
        st.session_state.pdbs.clear()
        st.success("Workspace cleared")

# -------------------------
# NETWORK tab
# -------------------------
elif section == "Network":
    st.header("Simple co-occurrence network from article abstracts (session)")
    docs = [a.get("abstract") or a.get("full_text") for a in st.session_state.articles]
    if not any(docs):
        st.info("No article texts in session. Save articles first.")
    else:
        G = build_cooccurrence_graph(docs, top_n_terms=40)
        plot_network(G)
        if st.button("Export network as JSON"):
            data = {"nodes":[{"id":n,"size":G.nodes[n].get("size",1)} for n in G.nodes()],
                    "links":[{"source":u,"target":v,"weight":G[u][v]["weight"]} for u,v in G.edges()]}
            st.download_button("Download network", data=json.dumps(data, indent=2).encode("utf-8"), file_name="network.json", mime="application/json")

# -------------------------
# Footer tips
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Tips: use the Workspace to collect items you want to export. Use the Network tab to explore term co-occurrence in saved article texts. Deploy to Streamlit Cloud by pushing this repo with `app.py` and `requirements.txt`.")
