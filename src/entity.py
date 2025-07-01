"""
Entity module for hierarchical pathfinding system.

This module contains the Entity class which handles movement and pathfinding
using a hierarchical approach with three levels:
- Level 2: Mega-chunk level pathfinding
- Level 1: Chunk level pathfinding  
- Level 0: Node level pathfinding

The entity uses line-of-sight checks for direct paths and falls back to
hierarchical pathfinding when needed.
"""

from data import Node
from utility import GraphAStar, get_node, NodeAStar, raycast
from settings import TILE_SIZE


class Entity:
    """
    An entity that can move around the map using hierarchical pathfinding.
    
    The entity uses a three-level hierarchical pathfinding system:
    1. Level 2 (Mega-chunk): High-level path between mega-chunks
    2. Level 1 (Chunk): Medium-level path within mega-chunks  
    3. Level 0 (Node): Low-level path between individual nodes
    
    The entity first checks for direct line-of-sight paths, then falls back
    to hierarchical pathfinding if needed.
    
    Attributes:
        x (float): Current x position in pixels
        y (float): Current y position in pixels
        speed (float): Movement speed in pixels per second
        level_2_path (list): Path at mega-chunk level (GraphNode objects)
        level_1_path (list): Path at chunk level (GraphNode objects)
        level_0_path (list): Path at node level (Node objects)
        target_node (Node): Current target node for movement
        goal_node (Node): Final destination node
    """
    
    def __init__(self, node_x: int, node_y: int, speed: float = 30):
        """
        Initialize the entity at a specific node position.
        
        Args:
            node_x (int): Initial x coordinate in node space
            node_y (int): Initial y coordinate in node space
            speed (float): Movement speed in pixels per second (default: 30)
        """
        self.x: float = node_x * TILE_SIZE
        self.y: float = node_y * TILE_SIZE
        self.speed = speed

        # Hierarchical pathfinding paths at different levels
        self.level_2_path = []  # Mega-chunk level path (Layer 2)
        self.level_1_path = []  # Chunk level path (Layer 1) 
        self.level_0_path = []  # Node level path (Layer 0)
        
        # Current target and final goal
        self.target_node = None
        self.goal_node = None

    @property
    def current_node(self) -> Node | None:
        """
        Get the node at the entity's current position.
        
        Returns:
            Node: The node at the current position, or None if invalid
        """
        node_x = round(self.x / TILE_SIZE)
        node_y = round(self.y / TILE_SIZE)
        node = get_node((node_x, node_y))
        return node

    def pathfind(self, end_node: Node) -> bool:
        """
        Find a path to the target node using hierarchical pathfinding.
        
        First checks for direct line-of-sight path. If that fails,
        uses hierarchical pathfinding starting from level 2 (mega-chunk).
        
        Args:
            end_node (Node): The target destination node
            
        Returns:
            bool: True if a path was found, False otherwise
        """
        if not end_node or not self.current_node:
            return False
            
        self.goal_node = end_node
        
        # First check if we can go directly to the target
        if self._check_direct_path(end_node):
            return True

        # Clear all existing paths
        self.level_2_path.clear()
        self.level_1_path.clear()
        self.level_0_path.clear()

        # Start hierarchical pathfinding at level 2 (mega-chunk)
        self.level_2_path = GraphAStar(self.current_node, end_node, layer=2)

        if not self.level_2_path:
            return False

        # Compute the next level down
        self.compute_next_level_1_path()
        return True

    def _check_direct_path(self, target_node: Node) -> bool:
        """
        Check if we can reach the target directly using line-of-sight and A*.
        
        Args:
            target_node (Node): The target node to check
            
        Returns:
            bool: True if direct path is possible, False otherwise
        """
        current_node = self.current_node
        if not current_node:
            return False
            
        # Check line-of-sight first
        raycast_valid = raycast(
            (current_node.x, current_node.y), (target_node.x, target_node.y)
        )
        
        if raycast_valid:
            # Verify with A* that a valid path actually exists
            direct_path = NodeAStar(current_node, target_node)
            if direct_path:
                # Set up direct movement
                self.target_node = target_node
                self.level_0_path.clear()
                self.level_1_path.clear()
                self.level_2_path.clear()
                return True
                
        return False

    def compute_next_level_1_path(self):
        """
        Compute the next chunk-level path from the mega-chunk path.
        
        Takes the next node from level_2_path and finds a chunk-level
        path to reach it. If successful, computes the next level 0 path.
        """
        if not self.level_2_path:
            return
            
        next_graph_node = self.level_2_path.pop(0)
        target_node = get_node((next_graph_node.x, next_graph_node.y))
        
        if not target_node:
            return

        if not self.current_node:
            return
        self.level_1_path = GraphAStar(self.current_node, target_node, layer=1)

        if self.level_1_path:
            self.compute_next_level_0_path()

    def compute_next_level_0_path(self):
        """
        Compute the next node-level path from the chunk path.
        
        Takes the next node from level_1_path and finds a node-level
        path to reach it. Sets the target_node for movement.
        """
        if not self.level_1_path:
            return
            
        next_graph_node = self.level_1_path.pop(0)
        target_node = get_node((next_graph_node.x, next_graph_node.y))
        
        if not target_node:
            return
        if not self.current_node:
            return
        self.level_0_path = NodeAStar(self.current_node, target_node)

        if self.level_0_path:
            # Remove current node if it's the first in path
            if self.level_0_path and self.level_0_path[0] == self.current_node:
                self.level_0_path.pop(0)
            self.target_node = self.level_0_path.pop(0) if self.level_0_path else target_node

    def update(self, dt: float):
        """
        Update the entity's state and movement.
        
        Handles movement towards the current target and manages
        the hierarchical pathfinding system.
        
        Args:
            dt (float): Delta time in seconds
        """
        if self.target_node:
            self.move_toward_target(dt)
        else:
            # No current target, try to get next target from paths
            if self.level_1_path:
                self.compute_next_level_0_path()
            elif self.level_2_path:
                self.compute_next_level_1_path()

    def move_toward_target(self, dt: float):
        """
        Move the entity towards the current target node.
        
        Handles movement logic and target switching when reaching
        waypoints in the path.
        
        Args:
            dt (float): Delta time in seconds
        """
        if not self.target_node:
            return

        # Calculate target position in pixels
        target_x = self.target_node.x * TILE_SIZE
        target_y = self.target_node.y * TILE_SIZE
        
        # Calculate direction vector
        dx = target_x - self.x
        dy = target_y - self.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance < 0.5:  # Close enough to target
            if self.target_node == self.goal_node:
                # Reached final destination
                return
                
            # Get next target from current path
            if self.level_0_path:
                self.target_node = self.level_0_path.pop(0)
            else:
                self.target_node = None
        else:
            # Move towards target
            if distance > 0:  # Avoid division by zero
                self.x += (dx / distance) * self.speed * dt
                self.y += (dy / distance) * self.speed * dt 