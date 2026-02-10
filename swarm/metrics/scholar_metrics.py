"""Metrics for scholar/literature synthesis domain."""

from typing import Any, Dict, List

from swarm.models.interaction import SoftInteraction


def citation_precision(interactions: List[SoftInteraction]) -> float:
    """Compute fraction of citations that faithfully support claims.

    A citation is considered faithful if its entailment score is above
    the threshold (typically 0.7).

    Args:
        interactions: List of completed interactions

    Returns:
        Precision score in [0, 1]
    """
    scholar_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "verify"
    ]

    if not scholar_interactions:
        return 1.0  # No citations to verify

    valid_count = sum(
        1 for i in scholar_interactions
        if i.metadata.get("actual_valid", False)
    )

    return valid_count / len(scholar_interactions)


def citation_recall(interactions: List[SoftInteraction]) -> float:
    """Compute fraction of gold citations found.

    Measures how many of the expected key points from queries
    were covered in synthesis results.

    Args:
        interactions: List of completed interactions

    Returns:
        Recall score in [0, 1]
    """
    synthesis_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "synthesize"
    ]

    if not synthesis_interactions:
        return 0.0

    total_coverage: float = sum(
        float(i.metadata.get("coverage", 0.0)) for i in synthesis_interactions
    )

    return total_coverage / len(synthesis_interactions)


def adversary_success_rate(interactions: List[SoftInteraction]) -> float:
    """Compute fraction of adversarial citations that survived verification.

    An adversarial citation "survives" if it was marked as valid during
    verification despite being adversarial.

    Args:
        interactions: List of completed interactions

    Returns:
        Success rate in [0, 1], lower is better for defenders
    """
    verify_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "verify"
    ]

    adversarial_citations = [
        i for i in verify_interactions
        if i.metadata.get("is_adversarial", False)
    ]

    if not adversarial_citations:
        return 0.0  # No adversarial citations

    survived = sum(
        1 for i in adversarial_citations
        if i.metadata.get("actual_valid", False)
    )

    return survived / len(adversarial_citations)


def contradiction_coverage(interactions: List[SoftInteraction]) -> float:
    """Compute fraction of known contradictions surfaced.

    Measures how well the synthesis process identifies and
    reports known contradictions in the literature.

    Args:
        interactions: List of completed interactions

    Returns:
        Coverage score in [0, 1]
    """
    synthesis_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "synthesize"
    ]

    if not synthesis_interactions:
        return 0.0

    # Count contradictions surfaced vs expected
    total_expected = 0
    total_surfaced = 0

    for i in synthesis_interactions:
        expected = len(i.metadata.get("known_contradictions", []))
        surfaced = len(i.metadata.get("contradictions_surfaced", []))
        total_expected += expected
        total_surfaced += surfaced

    if total_expected == 0:
        return 1.0  # No contradictions expected

    return min(1.0, total_surfaced / total_expected)


def hallucination_rate(interactions: List[SoftInteraction]) -> float:
    """Compute fraction of claims without supporting evidence.

    A hallucination is a claim that was cited but the citation
    was found to be invalid during verification.

    Args:
        interactions: List of completed interactions

    Returns:
        Hallucination rate in [0, 1], lower is better
    """
    verify_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "verify"
    ]

    if not verify_interactions:
        return 0.0

    invalid_citations = sum(
        1 for i in verify_interactions
        if not i.metadata.get("actual_valid", True)
    )

    return invalid_citations / len(verify_interactions)


def retrieval_quality(interactions: List[SoftInteraction]) -> float:
    """Compute average quality of retrieved passages.

    Based on relevance scores and coverage of query terms.

    Args:
        interactions: List of completed interactions

    Returns:
        Quality score in [0, 1]
    """
    retrieval_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "retrieve"
    ]

    if not retrieval_interactions:
        return 0.0

    total_quality = sum(
        i.task_progress_delta for i in retrieval_interactions
    )

    return total_quality / len(retrieval_interactions)


def verification_accuracy(interactions: List[SoftInteraction]) -> float:
    """Compute accuracy of verifier agents.

    Measures how often verifiers correctly identified valid vs invalid
    citations.

    Args:
        interactions: List of completed interactions

    Returns:
        Accuracy in [0, 1]
    """
    verify_interactions = [
        i for i in interactions
        if i.metadata.get("scholar") and i.metadata.get("action") == "verify"
    ]

    if not verify_interactions:
        return 1.0

    correct = sum(
        1 for i in verify_interactions
        if i.metadata.get("verdict") == i.metadata.get("actual_valid")
    )

    return correct / len(verify_interactions)


def scholar_metrics_summary(interactions: List[SoftInteraction]) -> Dict[str, Any]:
    """Compute all scholar metrics as a summary dict.

    Args:
        interactions: List of completed interactions

    Returns:
        Dictionary with all scholar metrics
    """
    return {
        "citation_precision": citation_precision(interactions),
        "citation_recall": citation_recall(interactions),
        "adversary_success_rate": adversary_success_rate(interactions),
        "contradiction_coverage": contradiction_coverage(interactions),
        "hallucination_rate": hallucination_rate(interactions),
        "retrieval_quality": retrieval_quality(interactions),
        "verification_accuracy": verification_accuracy(interactions),
    }
