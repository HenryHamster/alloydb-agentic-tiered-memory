"""Small graph helpers shared by consolidation and macro clustering."""

from collections.abc import Hashable, Iterable


def components(nodes: Iterable[Hashable], edges: Iterable[tuple[Hashable, Hashable]]) -> list[list]:
    """Connected components (union-find). Edges with an endpoint outside `nodes` are ignored.

    Each component is sorted, and components are ordered by their first member, so the result
    is deterministic for the same input.
    """
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left in parent and right in parent:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)
    groups: dict = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])
