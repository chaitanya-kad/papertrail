"""Streamlit interface for searching a local research library."""

from pathlib import Path

from .documents import Passage, load_documents, split_into_passages
from .retrieval import SearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = PROJECT_ROOT / "data" / "documents"
UPLOAD_LIBRARY = PROJECT_ROOT / "data" / "uploads"


def _cached_engine(passages: tuple[Passage, ...], use_semantic: bool) -> SearchEngine:
    """Cache the index and model across Streamlit reruns."""
    import streamlit as st

    @st.cache_resource(show_spinner="Preparing the search index…")
    def build(items: tuple[Passage, ...], semantic: bool) -> SearchEngine:
        engine = SearchEngine(list(items))
        if semantic and items:
            from .semantic import SemanticIndex

            engine.semantic = SemanticIndex(list(items))
        return engine

    return build(passages, use_semantic)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="PaperTrail · Research Search", page_icon="📚", layout="wide")
    st.title("📚 PaperTrail")
    st.caption("Find the passage you need. Compare keyword and meaning-based search.")

    with st.sidebar:
        st.subheader("Your library")
        st.write("Add Markdown, text, or PDF files here. PDFs keep page references.")
        uploads = st.file_uploader(
            "Add documents", type=["md", "txt", "pdf"], accept_multiple_files=True
        )
        if uploads:
            UPLOAD_LIBRARY.mkdir(parents=True, exist_ok=True)
            for upload in uploads:
                safe_name = Path(upload.name.replace("\\", "/")).name
                destination = UPLOAD_LIBRARY / safe_name
                if not destination.exists() or destination.read_bytes() != upload.getvalue():
                    destination.write_bytes(upload.getvalue())
            st.success(f"Added {len(uploads)} file(s) to data/uploads.")

        st.divider()
        st.caption("Sample papers are in `data/documents/`. Uploaded files are stored in `data/uploads/`.")

    try:
        documents = load_documents(DEFAULT_LIBRARY, source_prefix="sample")
        documents += load_documents(UPLOAD_LIBRARY, source_prefix="uploads")
        passages = split_into_passages(documents)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if not passages:
        st.info("Add a .md, .txt, or PDF file from the sidebar to begin searching.")
        st.stop()

    left, right = st.columns([3, 1])
    with left:
        query = st.text_input("Search your papers", placeholder="e.g. how do we measure retrieval quality?")
    with right:
        mode = st.selectbox("Search method", ["hybrid", "keyword", "semantic"])
    top_k = st.slider("Number of passages", min_value=1, max_value=10, value=5)

    if query.strip():
        try:
            engine = _cached_engine(tuple(passages), mode != "keyword")
            results = engine.search(query, mode=mode, top_k=top_k)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        st.subheader(f"Results ({len(results)})")
        if not results:
            st.info("No matching passages found. Try a different query or method.")
        for result in results:
            passage = result.passage
            location = passage.source
            if passage.page is not None:
                location += f" · page {passage.page}"
            st.markdown(f"**{location}** · passage {passage.number} · score `{result.score:.4f}`")
            st.write(passage.text)
            st.divider()

    with st.expander("Compare retrieval methods on the included benchmark"):
        st.write(
            "The benchmark uses a small hand-labeled set of sample queries. "
            "It is a learning aid, not a claim that one method wins on every collection."
        )
        if st.button("Run benchmark", type="secondary"):
            try:
                from .evaluation import compare_modes, load_queries

                queries = load_queries(PROJECT_ROOT / "data" / "evaluation" / "queries.json")
                with st.spinner("Loading the model and running the benchmark…"):
                    scores = compare_modes(passages, queries)
                st.dataframe(scores, use_container_width=True)
            except RuntimeError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
