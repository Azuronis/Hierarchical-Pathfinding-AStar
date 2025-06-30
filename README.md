# Hierarchical Pathfinding A*

An optimized pathfinding implementation that uses hierarchical pathfinding over traditional A* pathfinding on large maps.

## Overview

This project demonstrates hierarchical pathfinding - an advanced optimization technique that divides large maps into manageable chunks connected through entrance nodes. Instead of calculating paths across entire maps, the algorithm creates an abstract graph structure that dramatically reduces computational complexity and calcualted the path as the actor / entity moves.

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

## References

This implementation is based on established research in hierarchical pathfinding:

1. **[Near Optimal Hierarchical Pathfinding (HPA*)](https://webdocs.cs.ualberta.ca/~mmueller/ps/hpastar.pdf)** - Foundational algorithm for efficient pathfinding in large maps
2. **[Hierarchical Dynamic Pathfinding](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/Alain_Benoit_HierarchicalDynamicPathfinding.pdf)** - Application in dynamic environments

## Getting Started

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


**Performance tested on maps with varying complexity - results may vary based on map structure and unit movement speed.*
