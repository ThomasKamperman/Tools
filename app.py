"""
Streamlit multi-DB research assistant with PubChem + 3D viewers.
Features added:
 - PubChem compound search (2D PNG + 3D SDF)
 - Protein 3D viewer (PDB file from RCSB)
 - Uses 3Dmol.js embedded via st.components.html for interactive 3D
 - Safer error handling to avoid blank pages
"""

import os
import socket
import io
import json
from typing import List, Dict, Optional
from collections import Counter

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from xml.etree import ElementTree as ET
from Bio import SeqIO  # for robust FASTA handling


# Streamlit config
st.set_page_config(page_title="NCBI Research Assistant + PubChem + 3D", layout="wide")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEFAULT_DBS = ["pmc", "pubmed", "protein", "nucleotide"]

# ------ Helpers: HTTP with optional API key and safe requests ------
def ncbi_request(path: str, params: dict, timeout: int = 20) -> requests.Response:
    api_key = st.session_state.get("ncbi_api_key") or os.environ.get("NCBI_API_KEY")
    params = params.copy()
    if api_key:
        params["api_key"] = api_key
    url = BASE_URL + path
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp

# PubChem pug rest base
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# RCSB PDB files (raw)
RCSB_PDB_URL = "https://files.rcsb.org/download/"

# ------ Safe XML helpers ------
def safe_findtext(elem: ET.Element, path: str) -> Optional[str]:
    t = elem.findtext(path)
    return t.strip() if isinstance(t, str) else None

# ------ Search / fetchers (NCBI) ------
@st.cache_data(show_spinner=False)
def search_ids(query: str, db: str = "pubmed", max_results: int = 5) -> List[str]:
    try:
        params = {"db": db, "term": query, "retmax": max_results, "retmode": "json"}
        r = ncbi_request("esearch.fcgi", params)
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        st.error(f"Search error for db={db}: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_pubmed(pmid: str, full_article: bool = False, abstract_max_len: int = 300) -> Dict:
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = ncbi_request("efetch.fcgi", params)
        root = ET.fromstring(r.content)
        article = root.find(".//PubmedArticle")
        if article is None:
            return {"db": "pubmed", "id": pmid, "error": "No article found"}

        title = safe_findtext(article, ".//ArticleTitle") or "No title"
        year = safe_findtext(article, ".//JournalIssue/PubDate/Year")
        journal = safe_findtext(article, ".//Journal/Title")
        abstract_parts = [t.text for t in article.findall(".//AbstractText") if t.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract"

        if not full_article and len(abstract) > abstract_max_len:
            abstract = abstract[:abstract_max_len] + "..."

        return {"db":"pubmed","id":pmid,"title":title,"year":year,"journal":journal,"abstract":None if full_article else abstract,"full_text":abstract if full_article else None}
    except Exception as e:
        return {"db":"pubmed","id":pmid,"error":str(e)}

@st.cache_data(show_spinner=False)
def fetch_pmc(pmcid: str, full_article: bool = False, abstract_max_len: int = 300) -> Dict:
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
            return {"db":"pmc","id":pmcid,"title":title,"year":None,"journal":None,"abstract":None,"full_text": full_text or "Full text not available"}
        else:
            preview = (full_text[:abstract_max_len] + "...") if full_text and len(full_text) > abstract_max_len else (full_text or "No preview")
            return {"db":"pmc","id":pmcid,"title":title,"year":None,"journal":None,"abstract":preview,"full_text":None}
    except Exception as e:
        return {"db":"pmc","id":pmcid,"error":str(e)}

@st.cache_data(show_spinner=False)
def fetch_sequence(seq_id: str, db: str = "protein") -> Dict:
    try:
        params = {"db": db, "id": seq_id, "rettype": "fasta", "retmode": "text"}
        resp = ncbi_request("efetch.fcgi", params)
        fasta_text = resp.text.strip()
        return {"db":db,"id":seq_id,"fasta":fasta_text}
    except Exception as e:
        return {"db":db,"id":seq_id,"error":str(e)}

# ------ PubChem: search + fetch images / SDF 3D ------
def pubchem_search_name_or_cid(query: str) -> Optional[int]:
    """
    Try: if query is integer -> treat as CID.
    Otherwise search by name for first CID.
    """
    q = query.strip()
    try:
        if q.isdigit():
            return int(q)
        # fetch CID by name
        url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.requote_uri(q)}/cids/JSON"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if cids:
            return int(cids[0])
    except Exception as e:
        st.warning(f"PubChem search failed: {e}")
    return None

def pubchem_get_2d_png_bytes(cid: int) -> Optional[bytes]:
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/PNG"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.warning(f"PubChem PNG fetch error: {e}")
        return None

def pubchem_get_3d_sdf(cid: int) -> Optional[str]:
    """
    Request 3D SDF (record_type=3d) text for a CID.
    """
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type=3d"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.warning(f"PubChem 3D SDF fetch error: {e}")
        return None

# ------ PDB (RCSB) fetch (PDB file) ------
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

# ------ Visualization via 3Dmol.js embedded HTML ------
def make_3dmol_viewer_from_sdf(sdf_text: str, width: int = 600, height: int = 400):
    """
    Return HTML snippet embedding 3Dmol.js and loading SDF string.
    """
    # Escape JS string
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

def make_3dmol_viewer_from_pdb(pdb_text: str, width: int = 700, height: int = 500):
    pdb_js = json.dumps(pdb_text)
    html = f"""
    <div id="viewer" style="width:{width}px; height:{height}px; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
      const pdb = {pdb_js};
      viewer.addModel(pdb, "pdb");
      viewer.setStyle({{}}, {{cartoon:{{color: 'spectrum'}}}});
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.7}});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    return html

# ------ Plot helpers for sequences ------
def plot_protein_composition_inline(fasta_str: str):
    seq = "".join([line.strip() for line in fasta_str.splitlines() if not line.startswith(">")])
    counts = Counter(seq)
    if not counts:
        st.info("No sequence for plotting.")
        return
    aa = sorted(counts.keys())
    freq = [counts[a] for a in aa]
    fig, ax = plt.subplots(figsize=(9,3))
    ax.bar(aa, freq)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Count")
    ax.set_title("Amino Acid Composition")
    st.pyplot(fig)

def plot_nucleotide_gc_and_length_inline(fasta_str: str):
    seq = "".join([line.strip() for line in fasta_str.splitlines() if not line.startswith(">")]).upper()
    if not seq:
        st.info("No sequence for plotting.")
        return
    length = len(seq)
    gc = (seq.count("G")+seq.count("C"))/length*100 if length>0 else 0
    fig, ax = plt.subplots(1,2,figsize=(9,3))
    ax[0].bar(["GC%","AT%"], [gc, 100-gc])
    ax[0].set_ylim(0,100)
    ax[0].set_title("GC Content")
    ax[1].bar(["Length"], [length])
    ax[1].set_title("Sequence Length")
    st.pyplot(fig)

# ------ High level orchestration (search + display) ------
def best_search(query: str, databases: List[str], max_results: int, include_seq: bool,
                full_article: bool, abstract_max_len: int):
    results = []
    for db in databases:
        ids = search_ids(query, db=db, max_results=max_results)
        for rid in ids:
            try:
                if db == "pmc":
                    rec = fetch_pmc(rid, full_article=full_article, abstract_max_len=abstract_max_len)
                elif db == "pubmed":
                    rec = fetch_pubmed(rid, full_article=full_article, abstract_max_len=abstract_max_len)
                elif db in ("protein","nucleotide") and include_seq:
                    rec = fetch_sequence(rid, db=db)
                else:
                    rec = {"db":db,"id":rid}
                results.append(rec)
            except Exception as e:
                results.append({"db":db,"id":rid,"error":str(e)})
    return results

def fetch_by_id(db: str, record_id: str, full_article: bool=False, abstract_max_len: int=300):
    if db == "pmc":
        return fetch_pmc(record_id, full_article=full_article, abstract_max_len=abstract_max_len)
    if db == "pubmed":
        return fetch_pubmed(record_id, full_article=full_article, abstract_max_len=abstract_max_len)
    if db in ("protein","nucleotide"):
        return fetch_sequence(record_id, db=db)
    return {"db":db,"id":record_id,"error":"unsupported db"}

# ------ UI ------
st.title("Search Tool")
st.markdown("Search PubMed/PMC and sequences, preview/export, and view molecules (PubChem) or protein structures (PDB).")

# Sidebar controls
st.sidebar.header("Search & display options")
query = st.sidebar.text_input("Search query or ID", value="pancreas organ on a chip")
is_id_mode = st.sidebar.checkbox("Treat input as ID (fetch by ID)", value=False)
databases = st.sidebar.multiselect("Databases to search", options=DEFAULT_DBS, default=DEFAULT_DBS)
max_results = st.sidebar.number_input("Max results per DB", min_value=1, max_value=20, value=3)
include_seq = st.sidebar.checkbox("Include sequences", value=True)
full_article = st.sidebar.checkbox("Request full article when possible (PMC)", value=False)
abstract_max_len = st.sidebar.number_input("Abstract/preview max length", min_value=50, max_value=5000, value=400, step=50)
include_visuals = st.sidebar.checkbox("Plot sequence visuals", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("PubChem: search compound by name or CID and show 2D + 3D viewer.")
pubchem_query = st.sidebar.text_input("PubChem query (name or CID)", value="glucose")
st.sidebar.markdown("---")
st.sidebar.markdown("Protein 3D: provide a PDB ID (e.g., 1TUP) to visualize structure.")
pdb_input = st.sidebar.text_input("PDB ID (optional)", value="")

# API key input
st.sidebar.markdown("---")
ncbi_api_key_input = st.sidebar.text_input("NCBI API key (optional)", value="")
if ncbi_api_key_input:
    st.session_state["ncbi_api_key"] = ncbi_api_key_input

# Action buttons
run_label = "Fetch by ID" if is_id_mode else "Search"
if st.sidebar.button(run_label):
    if is_id_mode:
        db_choice = st.sidebar.selectbox("DB for ID", options=DEFAULT_DBS, index=0)
        if not query.strip():
            st.error("Provide an ID")
            results = []
        else:
            with st.spinner("Fetching by ID..."):
                results = [fetch_by_id(db_choice, query.strip(), full_article=full_article, abstract_max_len=abstract_max_len)]
    else:
        if not query.strip():
            st.error("Provide a search query")
            results = []
        else:
            with st.spinner("Searching multiple DBs..."):
                results = best_search(query.strip(), databases, max_results, include_seq, full_article, abstract_max_len)

    if not results:
        st.info("No results returned.")
    else:
        df_rows = []
        for r in results:
            if r.get("db") in ("pubmed","pmc"):
                df_rows.append({
                    "db": r.get("db"),
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "year": r.get("year"),
                    "journal": r.get("journal"),
                    "has_full_text": bool(r.get("full_text"))
                })
            elif r.get("db") in ("protein","nucleotide"):
                seq_preview = "\n".join((r.get("fasta","").splitlines()[0:5]))
                df_rows.append({"db": r.get("db"), "id": r.get("id"), "title": None, "sequence_preview": seq_preview})
            else:
                df_rows.append({"db": r.get("db"), "id": r.get("id")})
        df = pd.DataFrame(df_rows)
        st.subheader("Summary")
        st.dataframe(df, use_container_width=True)

        # Export
        st.markdown("### Export results")
        st.download_button("Export JSON", data=json.dumps(results, indent=2), file_name="ncbi_results.json", mime="application/json")
        st.download_button("Export CSV (summary)", data=df.to_csv(index=False).encode("utf-8"), file_name="ncbi_summary.csv", mime="text/csv")

        # Details
        st.markdown("---")
        st.subheader("Detailed results")
        for rec in results:
            with st.expander(f"{rec.get('db', 'unknown').upper()} — {rec.get('id')} — {rec.get('title','')}"):
                st.write("DB:", rec.get("db"))
                st.write("ID:", rec.get("id"))
                if rec.get("db") in ("pubmed","pmc"):
                    st.write("Title:", rec.get("title"))
                    if rec.get("year"):
                        st.write("Year:", rec.get("year"))
                    if rec.get("journal"):
                        st.write("Journal:", rec.get("journal"))
                    if rec.get("full_text"):
                        st.markdown("**Full text**")
                        st.text_area("Full text", value=rec.get("full_text"), height=300)
                        st.download_button("Download full text", data=rec.get("full_text",""), file_name=f"{rec.get('id')}_fulltext.txt")
                    elif rec.get("abstract"):
                        st.markdown("**Abstract / preview**")
                        st.write(rec.get("abstract"))
                elif rec.get("db") in ("protein","nucleotide"):
                    fasta = rec.get("fasta","")
                    if fasta:
                        st.markdown("FASTA (preview)")
                        st.code("\n".join(fasta.splitlines()[:30]))
                        st.download_button("Download FASTA", data=fasta, file_name=f"{rec.get('id')}.fasta", mime="text/fasta")
                        if include_visuals:
                            if rec.get("db") == "protein":
                                st.markdown("Protein: amino acid composition")
                                plot_protein_composition_inline(fasta)
                            else:
                                st.markdown("Nucleotide: GC% & length")
                                plot_nucleotide_gc_and_length_inline(fasta)
                else:
                    st.write(rec)

# ------ PubChem widget (separate, immediate) ------
st.markdown("---")
st.subheader("PubChem compound lookup (2D + 3D)")
with st.expander("PubChem compound search & viewers", expanded=True):
    pc_q = pubchem_query.strip()
    if st.button("Lookup PubChem compound (2D + 3D)"):
        if not pc_q:
            st.error("Enter a PubChem name or CID in the sidebar")
        else:
            with st.spinner("Searching PubChem..."):
                cid = pubchem_search_name_or_cid(pc_q)
                if not cid:
                    st.error("No PubChem CID found for that query.")
                else:
                    st.success(f"Found CID: {cid}")
                    png = pubchem_get_2d_png_bytes(cid)
                    if png:
                        st.image(png, caption=f"PubChem 2D rendering (CID {cid})")
                    sdf = pubchem_get_3d_sdf(cid)
                    if sdf:
                        st.markdown("**3D (interactive)**")
                        html = make_3dmol_viewer_from_sdf(sdf, width=700, height=450)
                        st.components.v1.html(html, height=480, scrolling=False)
                    else:
                        st.warning("No 3D SDF available for this compound.")

# ------ PDB viewer widget (separate) ------
st.markdown("---")
st.subheader("Protein 3D viewer (RCSB PDB)")
with st.expander("PDB viewer (enter PDB ID)", expanded=True):
    pdbid = pdb_input.strip()
    if st.button("Load PDB structure"):
        if not pdbid:
            st.error("Enter a PDB ID in the sidebar")
        else:
            with st.spinner(f"Fetching PDB {pdbid}..."):
                pdb_text = fetch_pdb_file(pdbid)
                if not pdb_text:
                    st.error(f"Could not load PDB {pdbid}")
                else:
                    st.success(f"PDB {pdbid} loaded")
                    html = make_3dmol_viewer_from_pdb(pdb_text, width=800, height=500)
                    st.components.v1.html(html, height=540, scrolling=False)

# Footer tips
st.markdown("---")
st.markdown("""
**Notes & tips**
- For molecule 3D: PubChem 3D SDF is available for many compounds but not all.
- For protein 3D: enter an RCSB PDB ID (not a UniProt ID) — e.g., 1TUP, 6LU7.
- If the page ever shows blank: check the terminal where `streamlit run app.py` is executing for tracebacks.
""")
