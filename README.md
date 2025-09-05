# v1.6.2
# WORK IN PROGRESS
- *This is a work in progress version. There may be some bugs or functionalites not included from v1.*

## Controls

- **WASD**: Pan around the map
- **1**: Switch to level 0 view
- **2**: Switch to level 1 view
- **3**: Switch to level 2 view
- **4**: Switch to level 3 view
- **5**: Switch to level 4 view
- *Will Add a scroll wheel or dropdown menu later.*

- *Constants for Pan Speed and others can be changed in settings.py*

## Technical Implementation


### Hierarchical Structure
- **Chunk**: 11×11 tile regions containing abstractions of connected areas
- **Abstraction**: Connected walkable regions within chunks (via flood-fill)
- **Graph Node**: Entrance points enabling inter-chunk pathfinding


## References
This implementation is based on established research in hierarchical pathfinding:

1. **[Near Optimal Hierarchical Pathfinding (HPA*)](https://webdocs.cs.ualberta.ca/~mmueller/ps/hpastar.pdf)** - Foundational algorithm for efficient pathfinding in large maps
2. **[Hierarchical Dynamic Pathfinding](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/Alain_Benoit_HierarchicalDynamicPathfinding.pdf)** - Application in dynamic environments
3. **[hierarchical-clearance](https://harablog.wordpress.com/2009/02/05/hierarchical-clearance-based-pathfinding/)** - Variable sized entrance spacing for certain sized entities. 


## What this v2 will include. 
- Segmented Flow Field Pathfinding
- Path refinment
- Better memory control for surfaces and rendering
- Better memory for node storage. (No more OOP for them)
- Dynamic Chunk Loading and Creation

https://github.com/user-attachments/assets/4912f17a-b9a8-4ad0-b2eb-fc1107ea0399


## 📄 License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

**Performance tested on maps with varying complexity - results may vary based on map structure and unit movement speed.*
