"""Social Simulacra integration for SWARM-Concordia bridge.

Implements the three-step prompt chain from Park et al. (2022):
1. Persona expansion: seed personas -> population via few-shot prompting
2. Top-level post generation: persona + community context -> posts
3. Reply generation: probabilistic reply chains with persona mixing

Also provides:
- WhatIf counterfactual injection (insert arbitrary personas into threads)
- Community configuration (goals, rules, norms)

Reference: Park et al., "Social Simulacra: Creating Populated Prototypes
for Social Computing Systems" (UIST 2022).
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from swarm.bridges.concordia.events import JudgeScores

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PersonaSeed:
    """A designer-provided seed persona (typically ~10 per community)."""

    name: str
    description: str
    community_role: str = ""

    def to_prompt_line(self) -> str:
        """Format as a few-shot example line for persona expansion."""
        role_part = f" ({self.community_role})" if self.community_role else ""
        return f"{self.name}{role_part}: {self.description}"


@dataclass
class ExpandedPersona:
    """A generated persona expanded from seeds."""

    persona_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    community_role: str = ""
    generated: bool = True
    seed_origin: str = ""  # which seed inspired this persona

    def to_prompt_context(self) -> str:
        """Format for embedding in post/reply generation prompts."""
        role_part = f" ({self.community_role})" if self.community_role else ""
        return f"{self.name}{role_part}: {self.description}"


@dataclass
class CommunityConfig:
    """Community description, goals, and rules for simulacra generation."""

    name: str = "Unnamed Community"
    description: str = ""
    goal: str = ""
    rules: List[str] = field(default_factory=list)
    norms: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format community context for prompt embedding."""
        parts = [f"Community: {self.name}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.rules:
            parts.append("Rules:\n" + "\n".join(f"- {r}" for r in self.rules))
        if self.norms:
            parts.append("Norms:\n" + "\n".join(f"- {n}" for n in self.norms))
        return "\n".join(parts)


@dataclass
class Post:
    """A top-level post or reply in a thread."""

    post_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    author: ExpandedPersona = field(default_factory=ExpandedPersona)
    content: str = ""
    parent_id: Optional[str] = None  # None = top-level post
    depth: int = 0

    @property
    def is_reply(self) -> bool:
        return self.parent_id is not None


@dataclass
class Thread:
    """A complete thread: one top-level post + replies."""

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    posts: List[Post] = field(default_factory=list)

    @property
    def root(self) -> Optional[Post]:
        return self.posts[0] if self.posts else None

    @property
    def replies(self) -> List[Post]:
        return [p for p in self.posts if p.is_reply]

    @property
    def participants(self) -> List[ExpandedPersona]:
        seen: dict[str, ExpandedPersona] = {}
        for post in self.posts:
            if post.author.persona_id not in seen:
                seen[post.author.persona_id] = post.author
        return list(seen.values())

    def to_narrative(self) -> str:
        """Convert thread to a narrative text block for SWARM judge scoring."""
        lines = []
        for post in self.posts:
            prefix = "  " * post.depth
            role = (
                f" ({post.author.community_role})"
                if post.author.community_role
                else ""
            )
            lines.append(f"{prefix}{post.author.name}{role}: {post.content}")
        return "\n".join(lines)


# Type for an LLM generation function: (prompt, temperature) -> response_text
LLMGenerateFn = Callable[[str, float], str]


# ---------------------------------------------------------------------------
# Persona Expander
# ---------------------------------------------------------------------------


class PersonaExpander:
    """Expand seed personas into a larger population via few-shot prompting.

    Follows Social Simulacra's Step 1: given ~10 seed personas, generate
    ~N thematically consistent new personas. Without an LLM client,
    produces deterministic synthetic personas for testing.
    """

    def __init__(
        self,
        community: CommunityConfig,
        seeds: List[PersonaSeed],
        *,
        llm_client: Optional[LLMGenerateFn] = None,
        temperature: float = 0.7,
    ):
        self._community = community
        self._seeds = seeds
        self._llm_client = llm_client
        self._temperature = temperature

    def expand(
        self,
        n: int = 20,
        *,
        rng: Optional[random.Random] = None,
    ) -> List[ExpandedPersona]:
        """Generate n new personas from seeds.

        Args:
            n: Number of personas to generate.
            rng: Optional seeded RNG for reproducibility.

        Returns:
            List of ExpandedPersona objects (seeds converted + new generated).
        """
        r = rng or random.Random()

        # Convert seeds to ExpandedPersona
        base_personas = [
            ExpandedPersona(
                name=seed.name,
                description=seed.description,
                community_role=seed.community_role,
                generated=False,
                seed_origin=seed.name,
            )
            for seed in self._seeds
        ]

        if self._llm_client is not None:
            generated = self._expand_via_llm(n, r)
        else:
            generated = self._expand_synthetic(n, r)

        return base_personas + generated

    def _expand_via_llm(
        self,
        n: int,
        rng: random.Random,
    ) -> List[ExpandedPersona]:
        """Expand using LLM few-shot prompting."""
        assert self._llm_client is not None

        prompt = self._build_expansion_prompt(n)
        response = self._llm_client(prompt, self._temperature)
        return self._parse_expanded_personas(response, rng)

    def _expand_synthetic(
        self,
        n: int,
        rng: random.Random,
    ) -> List[ExpandedPersona]:
        """Generate synthetic personas for testing (no LLM required)."""
        _ROLE_POOL = [
            "regular member",
            "newcomer",
            "lurker",
            "power user",
            "moderator candidate",
            "casual visitor",
            "expert contributor",
            "community organizer",
        ]
        _TRAIT_POOL = [
            "enthusiastic about the topic",
            "skeptical but curious",
            "deeply experienced",
            "learning the basics",
            "motivated by reputation",
            "focused on helping others",
            "here for entertainment",
            "interested in governance",
            "prone to disagreement",
            "values consensus",
        ]
        _NAME_PREFIXES = [
            "Alex", "Jordan", "Sam", "Morgan", "Casey",
            "Riley", "Taylor", "Quinn", "Avery", "Drew",
            "Kai", "Rowan", "Sage", "Phoenix", "River",
            "Dakota", "Emery", "Finley", "Harper", "Indigo",
        ]

        personas: List[ExpandedPersona] = []
        for i in range(n):
            seed = rng.choice(self._seeds)
            role = rng.choice(_ROLE_POOL)
            traits = rng.sample(_TRAIT_POOL, k=min(2, len(_TRAIT_POOL)))
            name_prefix = _NAME_PREFIXES[i % len(_NAME_PREFIXES)]
            suffix = i // len(_NAME_PREFIXES)
            name = f"{name_prefix}_{suffix}" if suffix > 0 else name_prefix

            description = (
                f"A {role} in {self._community.name} who is "
                f"{' and '.join(traits)}. "
                f"Inspired by the community presence of {seed.name}."
            )
            personas.append(
                ExpandedPersona(
                    name=name,
                    description=description,
                    community_role=role,
                    generated=True,
                    seed_origin=seed.name,
                )
            )

        return personas

    def _build_expansion_prompt(self, n: int) -> str:
        """Build the few-shot persona expansion prompt."""
        seed_lines = "\n".join(s.to_prompt_line() for s in self._seeds)
        return (
            f"{self._community.to_prompt_context()}\n\n"
            f"Here are some members of this community:\n"
            f"{seed_lines}\n\n"
            f"Generate {n} more diverse members of this community. "
            f"For each member, provide their name, role, and a brief "
            f"description of their personality and motivations. "
            f"Format each as: Name (role): description\n"
        )

    def _parse_expanded_personas(
        self,
        response: str,
        rng: random.Random,
    ) -> List[ExpandedPersona]:
        """Parse LLM response into ExpandedPersona objects."""
        personas: List[ExpandedPersona] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try to parse "Name (role): description" format
            colon_idx = line.find(":")
            if colon_idx == -1:
                continue
            header = line[:colon_idx].strip()
            description = line[colon_idx + 1 :].strip()

            # Extract role from parentheses if present
            name = header
            role = ""
            paren_start = header.find("(")
            paren_end = header.find(")")
            if paren_start != -1 and paren_end != -1:
                name = header[:paren_start].strip()
                role = header[paren_start + 1 : paren_end].strip()

            # Strip leading numbers/bullets
            for prefix in ["- ", "* ", ". "]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            # Strip "N. " pattern
            if len(name) > 2 and name[0].isdigit() and name[1] == ".":
                name = name[2:].strip()
            if len(name) > 3 and name[:2].isdigit() and name[2] == ".":
                name = name[3:].strip()

            if name and description:
                seed = rng.choice(self._seeds)
                personas.append(
                    ExpandedPersona(
                        name=name,
                        description=description,
                        community_role=role,
                        generated=True,
                        seed_origin=seed.name,
                    )
                )

        return personas


# ---------------------------------------------------------------------------
# Thread Generator
# ---------------------------------------------------------------------------


class ThreadGenerator:
    """Generate threads following Social Simulacra's reply mechanics.

    Reply probability: p ~ N(reply_mean, reply_stdev), coin-flipped per turn,
    capped at max_replies. New personas introduced with new_persona_prob
    probability vs. reusing existing thread participants.
    """

    def __init__(
        self,
        community: CommunityConfig,
        personas: List[ExpandedPersona],
        *,
        llm_client: Optional[LLMGenerateFn] = None,
        base_temperature: float = 0.7,
        reply_mean: float = 0.65,
        reply_stdev: float = 0.15,
        max_replies: int = 8,
        new_persona_prob: float = 0.5,
    ):
        self._community = community
        self._personas = personas
        self._llm_client = llm_client
        self._base_temperature = base_temperature
        self._reply_mean = reply_mean
        self._reply_stdev = reply_stdev
        self._max_replies = max_replies
        self._new_persona_prob = new_persona_prob

    def generate_thread(
        self,
        *,
        author: Optional[ExpandedPersona] = None,
        rng: Optional[random.Random] = None,
    ) -> Thread:
        """Generate a complete thread (top-level post + reply chain).

        Args:
            author: Optional specific author for the top-level post.
            rng: Optional seeded RNG for reproducibility.

        Returns:
            Thread with posts and replies.
        """
        r = rng or random.Random()

        if not self._personas:
            return Thread()

        # Step 1: Choose author and generate top-level post
        if author is None:
            author = r.choice(self._personas)

        root_content = self._generate_post_content(author, r)
        root = Post(
            author=author,
            content=root_content,
            depth=0,
        )

        thread = Thread(posts=[root])

        # Step 2: Generate reply chain
        # Sample reply probability from N(mean, stdev) per Social Simulacra
        reply_p = max(0.0, min(1.0, r.gauss(self._reply_mean, self._reply_stdev)))

        participants = [author]
        reply_count = 0
        current_parent = root

        while reply_count < self._max_replies:
            # Coin flip with sampled reply probability
            if r.random() > reply_p:
                break

            # Choose replier: new persona or existing participant
            if r.random() < self._new_persona_prob and len(self._personas) > 1:
                # Introduce new persona (not already in thread)
                available = [
                    p
                    for p in self._personas
                    if p.persona_id not in {pp.persona_id for pp in participants}
                ]
                if available:
                    replier = r.choice(available)
                else:
                    replier = r.choice(self._personas)
            else:
                replier = r.choice(participants)

            reply_content = self._generate_reply_content(
                replier, current_parent, thread, r
            )
            reply = Post(
                author=replier,
                content=reply_content,
                parent_id=current_parent.post_id,
                depth=min(current_parent.depth + 1, 5),  # cap nesting
            )
            thread.posts.append(reply)

            if replier.persona_id not in {p.persona_id for p in participants}:
                participants.append(replier)

            # Next reply can be to root or to any existing post
            current_parent = r.choice(thread.posts)
            reply_count += 1

        return thread

    def generate_threads(
        self,
        n: int,
        *,
        rng: Optional[random.Random] = None,
    ) -> List[Thread]:
        """Generate multiple threads."""
        r = rng or random.Random()
        return [self.generate_thread(rng=r) for _ in range(n)]

    def _generate_post_content(
        self,
        author: ExpandedPersona,
        rng: random.Random,
    ) -> str:
        """Generate top-level post content."""
        if self._llm_client is not None:
            prompt = (
                f"{self._community.to_prompt_context()}\n\n"
                f"You are {author.to_prompt_context()}\n\n"
                f"Write a post for this community. "
                f"Stay in character. Be concise (1-3 sentences)."
            )
            return self._llm_client(prompt, self._base_temperature)

        # Synthetic fallback
        _TEMPLATES = [
            "I've been thinking about {topic} and wanted to share my perspective.",
            "Quick question for the community about {topic}.",
            "Has anyone else noticed {observation}?",
            "Sharing something I learned about {topic} today.",
            "Looking for advice on {topic} from experienced members.",
        ]
        topic = self._community.name.lower()
        observation = f"changes in how we discuss {topic}"
        template = rng.choice(_TEMPLATES)
        return template.format(topic=topic, observation=observation)

    def _generate_reply_content(
        self,
        replier: ExpandedPersona,
        parent: Post,
        thread: Thread,
        rng: random.Random,
    ) -> str:
        """Generate reply content."""
        if self._llm_client is not None:
            prompt = (
                f"{self._community.to_prompt_context()}\n\n"
                f"Thread so far:\n{thread.to_narrative()}\n\n"
                f"You are {replier.to_prompt_context()}\n\n"
                f"Write a reply to {parent.author.name}'s post. "
                f"Stay in character. Be concise (1-2 sentences)."
            )
            return self._llm_client(prompt, self._base_temperature)

        # Synthetic fallback
        _REPLY_TEMPLATES = [
            "I agree with {parent_author}, and would add that this matters for our community.",
            "Interesting point, {parent_author}. From my experience, I see it differently.",
            "Thanks for bringing this up. I think we should discuss this more.",
            "Building on what {parent_author} said, I've noticed similar patterns.",
            "I'm not sure I follow, {parent_author}. Could you elaborate?",
            "This is exactly the kind of discussion we need here.",
            "Strong disagree. The evidence points the other way.",
            "+1 to this. Been wanting to say the same thing.",
        ]
        template = rng.choice(_REPLY_TEMPLATES)
        return template.format(parent_author=parent.author.name)


# ---------------------------------------------------------------------------
# WhatIf Counterfactual Injector
# ---------------------------------------------------------------------------


class WhatIfInjector:
    """Inject counterfactual personas into existing threads.

    Implements Social Simulacra's WhatIf feature: given an existing thread,
    inject a specified persona (e.g., "a troll", "a domain expert") and
    generate how the thread would evolve with this new participant.
    """

    def __init__(
        self,
        community: CommunityConfig,
        *,
        llm_client: Optional[LLMGenerateFn] = None,
        temperature: float = 0.7,
        max_injected_replies: int = 4,
    ):
        self._community = community
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_injected_replies = max_injected_replies

    def inject(
        self,
        thread: Thread,
        persona_description: str,
        *,
        persona_name: str = "Injected_Agent",
        rng: Optional[random.Random] = None,
    ) -> Tuple[Thread, ExpandedPersona]:
        """Inject a counterfactual persona into a thread.

        Args:
            thread: The existing thread to inject into.
            persona_description: Description of the persona to inject
                (e.g., "a troll who derails discussions").
            persona_name: Name for the injected persona.
            rng: Optional seeded RNG.

        Returns:
            Tuple of (new_thread_with_injection, injected_persona).
        """
        r = rng or random.Random()

        injected = ExpandedPersona(
            name=persona_name,
            description=persona_description,
            community_role="injected_counterfactual",
            generated=True,
            seed_origin="whatif",
        )

        # Build an evolving thread so each LLM prompt sees prior injected turns
        evolving_thread = Thread(
            thread_id=f"{thread.thread_id}_whatif_{injected.persona_id[:8]}",
            posts=list(thread.posts),
        )

        # Generate injected persona's reply to a random post
        target_post = r.choice(thread.posts)

        for i in range(self._max_injected_replies):
            reply_content = self._generate_injected_reply(
                injected, target_post, evolving_thread, r
            )
            reply = Post(
                author=injected,
                content=reply_content,
                parent_id=target_post.post_id,
                depth=min(target_post.depth + 1, 5),
            )
            evolving_thread.posts.append(reply)

            # Generate community response to the injection
            if thread.participants and i < self._max_injected_replies - 1:
                responder = r.choice(thread.participants)
                response_content = self._generate_community_response(
                    responder, reply, evolving_thread, r
                )
                response = Post(
                    author=responder,
                    content=response_content,
                    parent_id=reply.post_id,
                    depth=min(reply.depth + 1, 5),
                )
                evolving_thread.posts.append(response)
                target_post = response  # next injection replies to response

        return evolving_thread, injected

    def _generate_injected_reply(
        self,
        persona: ExpandedPersona,
        target: Post,
        thread: Thread,
        rng: random.Random,
    ) -> str:
        """Generate content for the injected persona's reply."""
        if self._llm_client is not None:
            prompt = (
                f"{self._community.to_prompt_context()}\n\n"
                f"Thread so far:\n{thread.to_narrative()}\n\n"
                f"You are {persona.to_prompt_context()}\n\n"
                f"Write a reply to {target.author.name}'s post. "
                f"Stay fully in character. Be concise (1-2 sentences)."
            )
            return self._llm_client(prompt, self._temperature)

        # Synthetic: role-based templates
        desc_lower = persona.description.lower()
        if "troll" in desc_lower:
            templates = [
                "Lol, this is completely wrong. You clearly don't understand anything.",
                "Imagine actually believing this. Embarrassing.",
                "This whole thread is a waste of time.",
            ]
        elif "expert" in desc_lower:
            templates = [
                "From a technical perspective, there are nuances being missed here.",
                "I've studied this extensively. The evidence suggests otherwise.",
                "Good discussion, but the literature points to a different conclusion.",
            ]
        elif "moderator" in desc_lower:
            templates = [
                "Let's keep this discussion civil and on-topic.",
                "This thread is getting heated. Please review the community guidelines.",
                "I'd like to redirect this conversation back to the original topic.",
            ]
        else:
            templates = [
                "Interesting thread. Here's my take on this.",
                "I'm new here, but I have some thoughts on this.",
                "Adding my perspective to this discussion.",
            ]
        return rng.choice(templates)

    def _generate_community_response(
        self,
        responder: ExpandedPersona,
        injected_post: Post,
        thread: Thread,
        rng: random.Random,
    ) -> str:
        """Generate an existing community member's response to the injection."""
        if self._llm_client is not None:
            prompt = (
                f"{self._community.to_prompt_context()}\n\n"
                f"Thread so far:\n{thread.to_narrative()}\n\n"
                f"A new participant ({injected_post.author.name}) just posted: "
                f'"{injected_post.content}"\n\n'
                f"You are {responder.to_prompt_context()}\n\n"
                f"Write a brief response to this new participant. "
                f"Stay in character. Be concise (1-2 sentences)."
            )
            return self._llm_client(prompt, self._temperature)

        # Synthetic fallback
        desc_lower = injected_post.author.description.lower()
        if "troll" in desc_lower:
            templates = [
                "Please keep the discussion constructive.",
                "I don't think that's a productive way to engage.",
                "Let's not escalate this.",
            ]
        else:
            templates = [
                f"Welcome, {injected_post.author.name}. Thanks for your input.",
                "That's a valid perspective. Let me think about that.",
                "Interesting point. How does that relate to the original question?",
            ]
        return rng.choice(templates)


# ---------------------------------------------------------------------------
# Thread-to-SWARM bridge helpers
# ---------------------------------------------------------------------------


def thread_to_narrative_samples(
    thread: Thread,
) -> List[Tuple[List[str], str]]:
    """Convert a thread into (agent_ids, narrative_text) pairs for SWARM scoring.

    Each post-reply pair is extracted as an interaction for SWARM to score.

    Returns:
        List of (agent_ids, narrative_text) tuples suitable for
        ConcordiaAdapter.process_narrative().
    """
    if len(thread.posts) < 2:
        if thread.posts:
            root = thread.posts[0]
            return [([root.author.name], root.content)]
        return []

    samples: List[Tuple[List[str], str]] = []

    # Build post lookup
    by_id = {p.post_id: p for p in thread.posts}

    for post in thread.posts:
        if post.parent_id is None:
            continue  # skip root unless paired
        parent = by_id.get(post.parent_id)
        if parent is None:
            continue

        agent_ids = [parent.author.name, post.author.name]
        narrative = (
            f"{parent.author.name}: {parent.content}\n"
            f"{post.author.name}: {post.content}"
        )
        samples.append((agent_ids, narrative))

    # If only root (no replies), return root as single-agent
    if not samples and thread.posts:
        root = thread.posts[0]
        samples.append(([root.author.name], root.content))

    return samples


def threads_to_judge_ground_truth(
    thread: Thread,
    *,
    troll_persona_ids: Optional[set[str]] = None,
) -> List[Tuple[List[str], str, JudgeScores]]:
    """Convert thread into (agent_ids, narrative, expected_scores) triples.

    When troll_persona_ids are specified, interactions involving trolls
    get high harm / low cooperation scores for ground-truth comparison.
    """
    troll_ids = troll_persona_ids or set()
    samples = thread_to_narrative_samples(thread)
    results: List[Tuple[List[str], str, JudgeScores]] = []

    for agent_ids, narrative in samples:
        # Check if any participant is a troll
        has_troll = any(aid in troll_ids for aid in agent_ids)

        if has_troll:
            scores = JudgeScores(
                progress=0.2,
                quality=0.2,
                cooperation=0.1,
                harm=0.7,
            )
        else:
            scores = JudgeScores(
                progress=0.6,
                quality=0.6,
                cooperation=0.7,
                harm=0.1,
            )

        results.append((agent_ids, narrative, scores))

    return results
