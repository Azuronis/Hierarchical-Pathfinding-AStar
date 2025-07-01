# Hierarchical Pathfinding A*

An optimized pathfinding implementation that uses hierarchical pathfinding over traditional A* pathfinding on large maps.

## Overview

This project demonstrates hierarchical pathfinding - an advanced optimization technique that divides large maps into manageable chunks connected through entrance nodes. Instead of calculating paths across entire maps, the algorithm creates an abstract graph structure that dramatically reduces computational complexity and calculates the path as the actor/entity moves.

## Performance

- **100x average performance increase** compared to standard A* pathfinding
- **150x-220x improvement** for long-distance pathfinding
- Optimized for large maps with numerous obstacles and multiple units

## Demo

### Basic Pathfinding Demo
*Single unit using Hierarchical Pathfinding with A* to navigate to random goal nodes*

https://github.com/user-attachments/assets/1f88272e-2b4c-4507-b8de-8b0e4e270f00

### Dynamic Map Editing
*Real-time map editing: creating and deleting walkable nodes with live pathfinding updates*

https://github.com/user-attachments/assets/b9557be9-0160-4bbf-9190-24ecd2e6f6b2

## Controls

- **Left Click**: Set start node for pathfinding
- **Right Click**: Set destination node for pathfinding  
- **Shift + Left Click/Drag**: Add walkable nodes
- **Shift + Right Click/Drag**: Remove walkable nodes
- **WASD**: Pan around the map
- **1**: Switch to chunk-level view
- **2**: Switch to mega-chunk level view
- *Constants for Pan Speed and others can be changed in settings.py*

## Technical Implementation

### Hierarchical Structure
- **Nodes**: Individual walkable tiles on the map
- **Chunks**: 11×11 tile regions containing abstractions of connected areas
- **Abstractions**: Connected walkable regions within chunks (via flood-fill)
- **Mega Chunks**: 3×3 chunk super-regions for higher-level optimization
- **Clusters**: Groups of connected abstractions across mega chunks
- **Graph Nodes**: Entrance points enabling inter-chunk pathfinding

### Algorithm Explanation

#### Initial Map Processing
The map is initially split into 11×11 chunks. Within each chunk, nodes are processed using flood-fill algorithms to identify connected regions called abstractions. Each abstraction represents a group of walkable nodes that can reach each other within the same chunk.

![Abstractions within chunks](https://github.com/user-attachments/assets/1dd96d9e-1bc9-410c-bbbb-fdcf0ef2ca18)

In the image above, you can see that some chunks contain multiple colored groups of nodes. Each color represents a different abstraction - nodes that are connected to each other but isolated from other groups within the same chunk.

If we were to color all separate abstractions within the same chunk uniformly, it would look like this:

![Unified chunk coloring](https://github.com/user-attachments/assets/4b49456d-2404-4859-a435-4b93b429f951)

All nodes within a chunk are stored in the chunk data structure. When nodes are created, they are only connected to neighboring nodes within the same chunk and abstraction. This prevents the internal A* algorithm from going out of bounds and maintains clean abstraction boundaries.

#### Entrance Generation
Once all chunks are processed, the system generates entrances between adjacent chunks. For each chunk abstraction, a border check examines all four sides for potential connections with neighboring chunks.

![Initial entrance candidates](https://github.com/user-attachments/assets/97906d20-d8a7-45cf-9bb6-c88a02ab4597)

Initially, the system identifies numerous potential entrances - every node that touches an adjacent chunk's border becomes a candidate. This creates an overwhelming number of connection points.

The cleanup process reduces these candidates to practical entrance points:

![Optimized entrances](https://github.com/user-attachments/assets/8464d29f-10fa-47f6-9b31-69ad1a5a854d)

After optimization, most chunk sides have fewer entrances. However, some chunks still maintain multiple entrances on a single side when the terrain creates natural forks or branches that require separate connection points for optimal pathfinding.

Entrances are created based on continuous lines of adjacent walkable cells between chunks. Each entrance group represents an uninterrupted passage between two chunk abstractions.

#### Connection Establishment
With entrances generated, the system creates connections within each chunk's abstractions. Since nodes know which abstraction they belong to, entrance nodes within the same abstraction can be connected to each other, forming internal pathfinding networks.

![Chunk connections](https://github.com/user-attachments/assets/ba27dff5-37da-44fd-9e26-fe8ec51ef8c7)

Adjacent chunk entrances are also connected when neighbor pair relationships are established. The optimal chunk configuration has four entrance nodes (one per side), though terrain complexity may require more.

#### Cluster Formation and Mega Chunks
After chunk connections are established, the algorithm creates a higher hierarchical level by grouping chunks into 3×3 mega chunks and forming clusters within them.

**Cluster Formation Process**:
1. **Entrance Collection**: All entrance nodes within a mega chunk are gathered
2. **Connection Analysis**: A flood-fill algorithm traces connections between abstractions through their entrance nodes
3. **Cluster Assignment**: Connected abstractions are grouped into clusters
4. **Isolation Handling**: Abstractions without entrances form single-abstraction clusters
   
![Cluster](https://github.com/user-attachments/assets/22efcf6d-86d3-4225-80f4-2b17e4b94278)

#### Mega Chunk Entrance Generation
Mega chunks require their own entrance system for connections to neighboring mega chunks:

**Border Analysis**: Each mega chunk examines borders with neighboring mega chunks, identifying where existing chunk entrances align.

**Entrance Optimization**: Rather than creating entrances for every connection:
- Groups of adjacent chunk entrances are analyzed
- The centroid (most central) entrance from each group becomes the mega chunk entrance
- This reduces high-level connections while maintaining pathfinding optimality

![image](https://github.com/user-attachments/assets/9e57b1f7-041a-40a1-9d1e-c4c89440cb8e)


**Connection Network**: Selected mega chunk entrances connect to counterparts in neighboring mega chunks, forming the highest hierarchical level.

### Three-Level Pathfinding Process

The system operates on three distinct hierarchical levels:

**Level 2 (Mega Chunk Level)**:
- Plans routes between mega chunks using cluster entrances
- Determines which mega chunks the path must traverse
- Provides highest-level route guidance

**Level 1 (Chunk Level)**:
- Plans routes between chunks within mega chunks using chunk entrances  
- Determines specific chunks and entrances within each mega chunk
- Bridges high-level and low-level planning

**Level 0 (Node Level)**:
- Uses traditional A* within individual chunks for exact node-to-node paths
- Calculates detailed paths only for the immediate area around the entity
- Recalculates as the entity moves through the hierarchy

![image](https://github.com/user-attachments/assets/54b65868-6df1-4ca2-8a91-918e195a7e37)


### Dynamic Recalculation

The pathfinding system dynamically recalculates portions of the path as needed:

**Trigger Conditions**:
- Entity reaches a new chunk
- Entity reaches a new mega chunk  
- Target destination changes
- Map modifications affect the current path

**Incremental Updates**: Rather than recalculating entire paths, the system maintains higher-level route planning and only recalculates the immediate detailed path as the entity progresses.

### Algorithm Flow
1. **Map Generation**: Perlin noise creates initial walkable areas
2. **Chunk Creation**: Map divided into manageable 11×11 tile chunks
3. **Flood Fill**: Each chunk analyzed to find connected walkable regions (abstractions)
4. **Entrance Generation**: Connection points placed between adjacent chunks
5. **Graph Construction**: Entrances linked to form hierarchical pathfinding graph
6. **Cluster Formation**: Higher-level groupings created for mega-chunk optimization

### Dynamic Updates
When nodes are added/removed, the system:
- Updates affected chunks via flood-fill recalculation
- Regenerates entrance connections between chunks
- Rebuilds cluster relationships in affected mega chunks
- Maintains graph consistency for continued pathfinding

## Use Cases

Perfect for:
- Large-scale strategy games with multiple units
- Real-time strategy (RTS) games requiring efficient pathfinding
- Open-world games with vast explorable areas
- Navigation systems managing many simultaneous entities
- Any application requiring optimized pathfinding on complex, large maps

## References

This implementation is based on established research in hierarchical pathfinding:

1. **[Near Optimal Hierarchical Pathfinding (HPA*)](https://webdocs.cs.ualberta.ca/~mmueller/ps/hpastar.pdf)** - Foundational algorithm for efficient pathfinding in large maps
2. **[Hierarchical Dynamic Pathfinding](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/Alain_Benoit_HierarchicalDynamicPathfinding.pdf)** - Application in dynamic environments

## Getting Started

### Prerequisites
- Python 3.7+
- Pygame-CE library

### Installation
```bash
git clone https://github.com/Azuronis/hierarchical-pathfinding-astar.git
cd hierarchical-pathfinding-astar
pip install pygame-ce
```

### Running the Demo
```bash
python main.py
```

### Usage
1. **Initial Setup**: The program generates a map using Perlin noise
2. **Set Pathfinding Points**: Left-click for start, right-click for destination
3. **Edit the Map**: Hold Shift + click/drag to add or remove walkable areas
4. **Navigate**: Use WASD to pan around the large map
5. **Switch Views**: Press 1 or 2 to toggle between visualization layers

## Why Hierarchical Pathfinding?

Traditional pathfinding algorithms become computationally expensive on large maps with many units. This hierarchical approach:
- Reduces search space complexity through abstraction
- Maintains near-optimal path quality
- Scales efficiently with map size
- Supports real-time applications with multiple entities
- Enables dynamic map modifications without full recalculation

---

## Coming soon
- Low-level flow feild pathfinding for large groups of entities within the same chunk.
- LOS pathfinding and path refinment for smoother pathfinding.
- UI for debugging and testing the program with over keyboard buttons.
- Possibly more layers, or dynamic layer creation based on the map size.

## 📄 License
This project is licensed under the MIT License - see the [LICENCE](./LICENCE.md) file for details.

**Performance tested on maps with varying complexity - results may vary based on map structure and unit movement speed.*
