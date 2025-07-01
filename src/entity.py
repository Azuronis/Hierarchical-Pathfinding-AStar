"""
Entity class for hierarchical pathfinding with line of sight optimization.

This module implements an entity that can navigate through the
system using three levels of abstraction:
- Level 0: Direct node-to-node pathfinding
- Level 1: Chunk-level pathfinding using abstractions
- Level 2: Mega-chunk level pathfinding using clusters
"""

from data import Node, GraphNode
from utility import GraphAStar, get_node, NodeAStar, raycast
from settings import TILE_SIZE
from typing import Optional


class Entity:
    """
    An Entity is an Actor or object that navigates. Holds the paths and is the actual thing that navigates.
    
    Attributes:
        x: Current X position in pixel coordinates
        y: Current Y position in pixel coordinates
        speed: Movement speed in pixels per second
        level_2_path: Path at mega-chunk level (clusters)
        level_1_path: Path at chunk level (abstractions)
        level_0_path: Path at node level (individual tiles)
        target_node: Current target node for movement
        goal_node: Final destination node
    """
    
    def __init__(self, node_x: int, node_y: int, speed: float = 30):
        """
        Initialize the entity at the specified node position.
        
        Args:
            node_x: Starting X coordinate in node space
            node_y: Starting Y coordinate in node space
            speed: Movement speed in pixels per second
        """
        self.x: float = node_x * TILE_SIZE
        self.y: float = node_y * TILE_SIZE
        self.speed = speed

        # Hierarchical path storage
        self.level_2_path: list[GraphNode] = []  # Mega-chunk path (Layer 2)
        self.level_1_path: list[GraphNode] = []  # Chunk path (Layer 1)
        self.level_0_path: list[Node] = []       # Node path (Layer 0)
        
        # Current navigation state
        self.target_node: Optional[Node] = None
        self.goal_node: Optional[Node] = None
    @property
    def current_node(self) -> Optional[Node]:
        """Get the node at the entity's current position."""
        node_x = round(self.x / TILE_SIZE)
        node_y = round(self.y / TILE_SIZE)
        return get_node((node_x, node_y))

    def pathfind(self, end_node: Node) -> bool:
        """
        Find a path to the specified end node using hierarchical pathfinding.
        
        This method implements the main pathfinding logic:
        1. Use hierarchical pathfinding through the three levels
        2. Validate and optimize the path
        
        Args:
            end_node: The destination node
            
        Returns:
            True if a path was found, False otherwise
        """
        if not end_node or not self.current_node:
            return False
            
        self.goal_node = end_node
        
        # Clear any existing paths
        self._clear_paths()
        
        # Use hierarchical pathfinding
        path_found = self._hierarchical_pathfind()
        
        # If no path was found at any level, clear the goal and return False
        if not path_found:
            self.goal_node = None
            self.target_node = None
            return False
            
        return True

    def _hierarchical_pathfind(self) -> bool:
        """
        Perform hierarchical pathfinding through all three levels.
        
        Returns:
            True if a path was found, False otherwise
        """
        if not self.current_node or not self.goal_node:
            return False
            
        # Level 2: Find path through clusters (mega-chunk level)
        self.level_2_path = GraphAStar(self.current_node, self.goal_node, layer=2)
        
        if not self.level_2_path:
            # Try level 1 as fallback
            self.level_1_path = GraphAStar(self.current_node, self.goal_node, layer=1)
            if not self.level_1_path:
                # Try level 0 as final fallback
                self.level_0_path = NodeAStar(self.current_node, self.goal_node)
                if not self.level_0_path:
                    return False
                else:
                    # Found a level 0 path
                    self.target_node = self.level_0_path[0] if self.level_0_path else self.goal_node
                    return True
            else:
                # Found a level 1 path
                self._compute_next_level_0_path()
                return True
            
        # Start computing the next level path
        self._compute_next_level_1_path()
        return True

    def _compute_next_level_1_path(self):
        """Compute the next level 1 path segment."""
        if not self.level_2_path:
            return
            
        # Get the next waypoint from level 2 path
        next_graph_node = self.level_2_path[0]
        target_node = get_node((next_graph_node.x, next_graph_node.y))
        
        if not target_node:
            return
            
        # Find level 1 path to this waypoint
        if not self.current_node:
            return
        self.level_1_path = GraphAStar(self.current_node, target_node, layer=1)
        
        if self.level_1_path:
            self._compute_next_level_0_path()
        else:
            # If we can't find a level 1 path, try the next level 2 waypoint
            self.level_2_path.pop(0)
            self._compute_next_level_1_path()

    def _compute_next_level_0_path(self):
        """Compute the next level 0 path segment."""
        if not self.level_1_path:
            return
            
        # Get the next waypoint from level 1 path
        next_graph_node = self.level_1_path[0]
        target_node = get_node((next_graph_node.x, next_graph_node.y))
        
        if not target_node:
            return
            
        # Find level 0 path to this waypoint
        if not self.current_node:
            return
        self.level_0_path = NodeAStar(self.current_node, target_node)
        
        if self.level_0_path:
            # Remove current node from path and set target
            if self.level_0_path and self.level_0_path[0] == self.current_node:
                self.level_0_path.pop(0)
            self.target_node = self.level_0_path[0] if self.level_0_path else target_node
        else:
            # If we can't find a level 0 path, try the next level 1 waypoint
            self.level_1_path.pop(0)
            self._compute_next_level_0_path()

    def _clear_paths(self):
        """Clear all path data."""
        self.level_2_path.clear()
        self.level_1_path.clear()
        self.level_0_path.clear()
        self.target_node = None

    def update(self, dt: float):
        """
        Update the entity's movement and pathfinding logic.
        
        This method handles:
        1. Movement towards the current target
        2. Path progression through the hierarchy
        3. Path validation and optimization
        
        Args:
            dt: Delta time in seconds
        """
        if not self.goal_node:
            return
            
        # Move towards current target
        if self.target_node:
            self._move_toward_target(dt)
        else:
            # Progress to next path segment
            self._progress_path()

    def _progress_path(self):
        """Progress through the hierarchical path."""
        # Try to get next level 0 path segment
        if self.level_0_path:
            self.target_node = self.level_0_path.pop(0)
        # Try to get next level 1 path segment
        elif self.level_1_path:
            self.level_1_path.pop(0)  # Remove the waypoint we just reached
            self._compute_next_level_0_path()
        # Try to get next level 2 path segment
        elif self.level_2_path:
            self.level_2_path.pop(0)  # Remove the waypoint we just reached
            self._compute_next_level_1_path()
        # If we have no more paths, we've reached the goal
        elif self.goal_node:
            self.target_node = self.goal_node

    def _move_toward_target(self, dt: float):
        """
        Move the entity towards the current target node.
        
        Args:
            dt: Delta time in seconds
        """
        if not self.target_node:
            return

        # Calculate target position in pixel coordinates
        target_x = self.target_node.x * TILE_SIZE
        target_y = self.target_node.y * TILE_SIZE
        
        # Calculate direction vector
        dx = target_x - self.x
        dy = target_y - self.y
        distance = (dx**2 + dy**2) ** 0.5

        # Check if we've reached the target (use a larger threshold to prevent stuttering)
        if distance < 5.0:  # Increased threshold for smoother movement
            # Snap to exact target position to prevent pixel drift
            self.x = target_x
            self.y = target_y
            
            # Check if we've reached the final goal
            if self.target_node == self.goal_node:
                self.target_node = None
                return
                
            # Move to next target in the path
            self.target_node = None
        else:
            # Move towards target
            if distance > 0:
                # Calculate movement step
                step = self.speed * dt
                
                # If we're close enough, move directly to target to prevent overshooting
                if distance <= step:
                    self.x = target_x
                    self.y = target_y
                else:
                    # Normal movement
                    self.x += (dx / distance) * step
                    self.y += (dy / distance) * step
