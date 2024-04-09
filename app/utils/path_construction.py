import networkx as nx


def create_graph_from_image(binary_image):
    rows, cols = binary_image.shape
    G = nx.grid_2d_graph(rows, cols, create_using=nx.Graph())

    # Add diagonal edges for Moore neighborhood connectivity
    for y in range(rows):
        for x in range(cols):
            # Diagonal directions
            for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < cols and 0 <= ny_ < rows:
                    # Before adding the edge, check if diagonal pass-throughs are walkable
                    if (
                        binary_image[y, x] and binary_image[ny_, nx_]
                    ):  # Check current and target for walkability
                        G.add_edge((y, x), (ny_, nx_))

    # Remove nodes and their edges for non-walkable paths
    # This also implicitly removes unwanted diagonal edges by removing nodes
    for y in range(rows):
        for x in range(cols):
            if not binary_image[y, x]:  # If the current node is non-walkable
                if (y, x) in G:
                    G.remove_node((y, x))

    # Remove diagonal edges if blocked by diagonal obstacles
    for y in range(rows - 1):
        for x in range(cols - 1):
            if not binary_image[y, x] and not binary_image[y + 1, x + 1]:
                # Remove edge if exists and is diagonal through blocked cells
                if G.has_edge((y, x + 1), (y + 1, x)):
                    G.remove_edge((y, x + 1), (y + 1, x))
                if G.has_edge((y + 1, x), (y, x + 1)):
                    G.remove_edge((y + 1, x), (y, x + 1))

            if not binary_image[y, x + 1] and not binary_image[y + 1, x]:
                # Remove edge if exists and is diagonal through blocked cells
                if G.has_edge((y, x), (y + 1, x + 1)):
                    G.remove_edge((y, x), (y + 1, x + 1))
                if G.has_edge((y + 1, x + 1), (y, x)):
                    G.remove_edge((y + 1, x + 1), (y, x))
    return G


def calculate_distance(G, node1, node2):
    # Use A* to find the shortest path and its length
    path_length = nx.astar_path_length(G, node1, node2)
    return path_length


def find_closest_node(G, current_node, nodes_to_visit):
    closest_node = None
    min_distance = float("inf")
    for node in nodes_to_visit:
        distance = calculate_distance(G, current_node, node)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node, min_distance


def tsp_heuristic(G, start_node, nodes_to_visit):
    visited_order = [start_node]
    current_node = start_node
    unvisited_nodes = set(nodes_to_visit)
    if start_node in unvisited_nodes:
        unvisited_nodes.remove(start_node)

    total_distance = 0

    while unvisited_nodes:
        next_node, distance = find_closest_node(G, current_node, unvisited_nodes)
        visited_order.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node
        total_distance += distance

    return visited_order, total_distance


def construct_path(G, visit_order):
    full_path = []
    for i in range(len(visit_order) - 1):
        start_node = visit_order[i]
        end_node = visit_order[i + 1]
        # Find the path between the current node and the next
        path = nx.astar_path(G, start_node, end_node)

        # Append the path to the full path
        # If not the first path, remove the first node to avoid duplication
        if i > 0:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)

    return full_path
