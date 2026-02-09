"""Search AgentXiv for papers on multi-agent safety topics."""

from swarm.research.platforms import AgentxivClient

TOPICS = [
    "multi-agent safety",
    "distributional welfare",
    "agent governance",
]


def main():
    client = AgentxivClient()  # no API key needed for search

    for topic in TOPICS:
        print(f"\n{'=' * 72}")
        print(f"  Search query: {topic!r}")
        print(f"{'=' * 72}")

        result = client.search(topic)

        if not result.papers:
            print(f"  No results found (total_count={result.total_count}).")
            continue

        print(f"  Total count: {result.total_count}")
        print(f"  Papers returned: {len(result.papers)}")

        for i, paper in enumerate(result.papers, 1):
            print(f"\n  --- Paper {i} ---")
            print(f"  paper_id : {paper.paper_id}")
            print(f"  title    : {paper.title}")
            print(f"  authors  : {', '.join(paper.authors) if paper.authors else '(none)'}")
            print(f"  upvotes  : {paper.upvotes}")
            abstract_text = paper.abstract.strip() if paper.abstract else "(no abstract)"
            # Indent the abstract for readability
            indented = "\n             ".join(abstract_text.splitlines())
            print(f"  abstract : {indented}")

    print(f"\n{'=' * 72}")
    print("  Done.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
