def get_connected_components(adj: dict[str, set[str]]):
    """
    Given a graph, returns the connected components as a list of sets of keys.
    """
    visited = set()
    components = []

    for v in adj:
        if v not in visited:
            comp = set()
            q = [v]
            while q:
                curr = q.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                comp.add(curr)
                q.extend(list(adj.get(curr, [])))

            components.append(comp)
    return components


def subdictionary(nodes, adj) -> dict[str, set[str]]:
    subdict = {}
    for node in nodes:
        subdict[node] = adj[node]
    return subdict


def is_path_graph(graph_dict: dict[str, set[str]]):
    """
    Checks if a graph is isomorphic to a path graph checking connectedness and the
    degree sequence.
    """
    n = len(graph_dict)

    if n == 0:
        return False
    if n == 1:
        return len(next(iter(graph_dict.values()))) == 0

    degrees = [len(neighbors) for neighbors in graph_dict.values()]

    if degrees.count(1) != 2 or degrees.count(2) != n - 2:
        return False

    visited = set()

    start_node = next(
        node for node, neighbors in graph_dict.items() if len(neighbors) == 1
    )

    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph_dict[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return len(visited) == n
