"""Feed engine for posts, comments, and voting."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class VoteType(Enum):
    """Types of votes."""

    UPVOTE = "upvote"
    DOWNVOTE = "downvote"


@dataclass
class Vote:
    """A vote on a post."""

    vote_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    post_id: str = ""
    voter_id: str = ""
    vote_type: VoteType = VoteType.UPVOTE
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Serialize vote."""
        return {
            "vote_id": self.vote_id,
            "post_id": self.post_id,
            "voter_id": self.voter_id,
            "vote_type": self.vote_type.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Post:
    """
    A post in the feed.

    Posts can be top-level or replies (via parent_id).
    """

    post_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    author_id: str = ""
    content: str = ""
    parent_id: Optional[str] = None  # None for top-level posts
    created_at: datetime = field(default_factory=datetime.now)

    # Engagement metrics (computed by Feed)
    upvotes: int = 0
    downvotes: int = 0
    reply_count: int = 0

    # Optional task association
    task_id: Optional[str] = None

    # Moderation status
    is_hidden: bool = False
    hidden_reason: Optional[str] = None

    @property
    def is_reply(self) -> bool:
        """Check if this is a reply."""
        return self.parent_id is not None

    @property
    def net_votes(self) -> int:
        """Net vote count (upvotes - downvotes)."""
        return self.upvotes - self.downvotes

    @property
    def age_seconds(self) -> float:
        """Age of post in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> Dict:
        """Serialize post."""
        return {
            "post_id": self.post_id,
            "author_id": self.author_id,
            "content": self.content,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "reply_count": self.reply_count,
            "task_id": self.task_id,
            "is_hidden": self.is_hidden,
            "hidden_reason": self.hidden_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Post":
        """Deserialize post."""
        return cls(
            post_id=data["post_id"],
            author_id=data["author_id"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            upvotes=data.get("upvotes", 0),
            downvotes=data.get("downvotes", 0),
            reply_count=data.get("reply_count", 0),
            task_id=data.get("task_id"),
            is_hidden=data.get("is_hidden", False),
            hidden_reason=data.get("hidden_reason"),
        )


class Feed:
    """
    Feed engine for managing posts and interactions.

    Implements visibility ranking: score = votes + α*replies - β*age
    """

    def __init__(
        self,
        reply_weight: float = 0.5,
        age_decay: float = 0.01,
        max_content_length: int = 10000,
    ):
        """
        Initialize feed.

        Args:
            reply_weight: Weight for reply count in ranking (α)
            age_decay: Decay factor per hour for age penalty (β)
            max_content_length: Maximum post content length
        """
        self.reply_weight = reply_weight
        self.age_decay = age_decay
        self.max_content_length = max_content_length

        # Storage
        self._posts: Dict[str, Post] = {}
        self._votes: Dict[str, Vote] = {}  # vote_id -> Vote

        # Indexes
        self._posts_by_author: Dict[str, List[str]] = {}  # author_id -> [post_id]
        self._replies_by_parent: Dict[str, List[str]] = {}  # parent_id -> [post_id]
        self._votes_by_post: Dict[str, List[str]] = {}  # post_id -> [vote_id]
        self._votes_by_voter: Dict[
            str, Dict[str, str]
        ] = {}  # voter_id -> {post_id: vote_id}

    def create_post(
        self,
        author_id: str,
        content: str,
        parent_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Post:
        """
        Create a new post.

        Args:
            author_id: ID of the author
            content: Post content
            parent_id: ID of parent post (for replies)
            task_id: Associated task ID (optional)

        Returns:
            The created Post

        Raises:
            ValueError: If content too long or parent doesn't exist
        """
        if len(content) > self.max_content_length:
            raise ValueError(
                f"Content length {len(content)} exceeds maximum {self.max_content_length}"
            )

        if parent_id and parent_id not in self._posts:
            raise ValueError(f"Parent post {parent_id} does not exist")

        post = Post(
            author_id=author_id,
            content=content,
            parent_id=parent_id,
            task_id=task_id,
        )

        # Store post
        self._posts[post.post_id] = post

        # Update author index
        if author_id not in self._posts_by_author:
            self._posts_by_author[author_id] = []
        self._posts_by_author[author_id].append(post.post_id)

        # Update parent reply count
        if parent_id:
            if parent_id not in self._replies_by_parent:
                self._replies_by_parent[parent_id] = []
            self._replies_by_parent[parent_id].append(post.post_id)
            self._posts[parent_id].reply_count += 1

        return post

    def get_post(self, post_id: str) -> Optional[Post]:
        """Get a post by ID."""
        return self._posts.get(post_id)

    def get_posts_by_author(self, author_id: str) -> List[Post]:
        """Get all posts by an author."""
        post_ids = self._posts_by_author.get(author_id, [])
        return [self._posts[pid] for pid in post_ids if pid in self._posts]

    def get_replies(self, post_id: str) -> List[Post]:
        """Get all replies to a post."""
        reply_ids = self._replies_by_parent.get(post_id, [])
        return [self._posts[rid] for rid in reply_ids if rid in self._posts]

    def vote(
        self,
        post_id: str,
        voter_id: str,
        vote_type: VoteType,
    ) -> Optional[Vote]:
        """
        Cast a vote on a post.

        Args:
            post_id: ID of the post to vote on
            voter_id: ID of the voter
            vote_type: Type of vote

        Returns:
            The created Vote, or None if post doesn't exist
        """
        if post_id not in self._posts:
            return None

        post = self._posts[post_id]

        # Check if already voted
        if voter_id not in self._votes_by_voter:
            self._votes_by_voter[voter_id] = {}

        existing_vote_id = self._votes_by_voter[voter_id].get(post_id)

        if existing_vote_id:
            # Update existing vote
            existing_vote = self._votes[existing_vote_id]
            old_type = existing_vote.vote_type

            if old_type == vote_type:
                # Same vote, no change
                return existing_vote

            # Undo old vote
            if old_type == VoteType.UPVOTE:
                post.upvotes -= 1
            else:
                post.downvotes -= 1

            # Apply new vote
            existing_vote.vote_type = vote_type
            existing_vote.timestamp = datetime.now()

            if vote_type == VoteType.UPVOTE:
                post.upvotes += 1
            else:
                post.downvotes += 1

            return existing_vote

        # New vote
        vote = Vote(
            post_id=post_id,
            voter_id=voter_id,
            vote_type=vote_type,
        )

        self._votes[vote.vote_id] = vote

        # Update indexes
        if post_id not in self._votes_by_post:
            self._votes_by_post[post_id] = []
        self._votes_by_post[post_id].append(vote.vote_id)
        self._votes_by_voter[voter_id][post_id] = vote.vote_id

        # Update post vote counts
        if vote_type == VoteType.UPVOTE:
            post.upvotes += 1
        else:
            post.downvotes += 1

        return vote

    def remove_vote(self, post_id: str, voter_id: str) -> bool:
        """
        Remove a vote from a post.

        Returns:
            True if vote was removed, False if no vote existed
        """
        if voter_id not in self._votes_by_voter:
            return False

        vote_id = self._votes_by_voter[voter_id].get(post_id)
        if not vote_id:
            return False

        vote = self._votes[vote_id]
        post = self._posts.get(post_id)

        if post:
            if vote.vote_type == VoteType.UPVOTE:
                post.upvotes -= 1
            else:
                post.downvotes -= 1

        # Clean up indexes
        del self._votes[vote_id]
        del self._votes_by_voter[voter_id][post_id]
        if post_id in self._votes_by_post:
            self._votes_by_post[post_id].remove(vote_id)

        return True

    def compute_visibility_score(self, post: Post) -> float:
        """
        Compute visibility score for ranking.

        score = net_votes + α * replies - β * age_hours

        Args:
            post: The post to score

        Returns:
            Visibility score (higher = more visible)
        """
        if post.is_hidden:
            return float("-inf")

        age_hours = post.age_seconds / 3600.0
        score = (
            post.net_votes
            + self.reply_weight * post.reply_count
            - self.age_decay * age_hours
        )
        return score

    def get_ranked_posts(
        self,
        limit: int = 50,
        include_replies: bool = False,
        include_hidden: bool = False,
    ) -> List[Post]:
        """
        Get posts ranked by visibility score.

        Args:
            limit: Maximum number of posts to return
            include_replies: Whether to include reply posts
            include_hidden: Whether to include hidden posts

        Returns:
            List of posts sorted by visibility score (descending)
        """
        posts = []
        for post in self._posts.values():
            if not include_replies and post.is_reply:
                continue
            if not include_hidden and post.is_hidden:
                continue
            posts.append(post)

        # Sort by visibility score
        posts.sort(key=lambda p: self.compute_visibility_score(p), reverse=True)

        return posts[:limit]

    def get_thread(self, post_id: str, max_depth: int = 10) -> List[Post]:
        """
        Get a post and all its replies as a thread.

        Args:
            post_id: ID of the root post
            max_depth: Maximum reply depth to traverse

        Returns:
            List of posts in the thread (root first, then replies sorted by score)
        """
        root = self.get_post(post_id)
        if not root:
            return []

        result = [root]
        self._collect_replies(post_id, result, 0, max_depth)
        return result

    def _collect_replies(
        self,
        post_id: str,
        result: List[Post],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively collect replies."""
        if depth >= max_depth:
            return

        replies = self.get_replies(post_id)
        # Sort replies by score
        replies.sort(key=lambda p: self.compute_visibility_score(p), reverse=True)

        for reply in replies:
            result.append(reply)
            self._collect_replies(reply.post_id, result, depth + 1, max_depth)

    def hide_post(self, post_id: str, reason: str = "moderation") -> bool:
        """
        Hide a post (moderation action).

        Args:
            post_id: ID of the post to hide
            reason: Reason for hiding

        Returns:
            True if post was hidden, False if not found
        """
        post = self.get_post(post_id)
        if not post:
            return False

        post.is_hidden = True
        post.hidden_reason = reason
        return True

    def unhide_post(self, post_id: str) -> bool:
        """Unhide a post."""
        post = self.get_post(post_id)
        if not post:
            return False

        post.is_hidden = False
        post.hidden_reason = None
        return True

    def get_recent_posts(
        self,
        since: datetime,
        author_id: Optional[str] = None,
    ) -> List[Post]:
        """
        Get posts created after a given time.

        Args:
            since: Cutoff datetime
            author_id: Optional filter by author

        Returns:
            List of recent posts
        """
        posts = []
        for post in self._posts.values():
            if post.created_at < since:
                continue
            if author_id and post.author_id != author_id:
                continue
            posts.append(post)

        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts

    def get_engagement_stats(self, post_id: str) -> Dict:
        """Get engagement statistics for a post."""
        post = self.get_post(post_id)
        if not post:
            return {}

        return {
            "post_id": post_id,
            "upvotes": post.upvotes,
            "downvotes": post.downvotes,
            "net_votes": post.net_votes,
            "reply_count": post.reply_count,
            "visibility_score": self.compute_visibility_score(post),
            "age_hours": post.age_seconds / 3600.0,
        }

    def get_feed_stats(self) -> Dict:
        """Get overall feed statistics."""
        total_posts = len(self._posts)
        top_level = sum(1 for p in self._posts.values() if not p.is_reply)
        replies = total_posts - top_level
        hidden = sum(1 for p in self._posts.values() if p.is_hidden)

        total_upvotes = sum(p.upvotes for p in self._posts.values())
        total_downvotes = sum(p.downvotes for p in self._posts.values())

        return {
            "total_posts": total_posts,
            "top_level_posts": top_level,
            "replies": replies,
            "hidden_posts": hidden,
            "total_votes": len(self._votes),
            "total_upvotes": total_upvotes,
            "total_downvotes": total_downvotes,
            "unique_authors": len(self._posts_by_author),
            "unique_voters": len(self._votes_by_voter),
        }

    def clear(self) -> None:
        """Clear all feed data."""
        self._posts.clear()
        self._votes.clear()
        self._posts_by_author.clear()
        self._replies_by_parent.clear()
        self._votes_by_post.clear()
        self._votes_by_voter.clear()
