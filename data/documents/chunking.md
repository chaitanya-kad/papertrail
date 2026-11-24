# Splitting documents into passages

Long documents are often split into smaller passages before indexing. This makes retrieval more precise because a search result can point to the section that answers a question instead of returning an entire paper. Passage size is a trade-off: very small passages may lose context, while very large passages can include unrelated material.

Overlapping neighboring passages preserves some context near the split boundary. The amount of overlap should be measured rather than chosen blindly. A useful passage also needs metadata such as its source filename and position so the interface can show where it came from.
