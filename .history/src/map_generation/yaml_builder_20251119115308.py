"""
YAML Configuration Builder for IRSIM
Converts MapBuilder objects to IRSIM-compatible YAML configurations
"""

import numpy as np
from typing import Dict, List, Any
from map_generator import MapBuilder, Robot, Obstacle, ShapeType, MaterialType

class YAMLBuilder:
    """Builds IRSIM-compatible YAML configurations from MapBuilder objects"""

    def __init__(self, map_builder: MapBuilder):
        self.map_builder = map_builder
        self.yaml_config = {}

    def build_yaml(self) -> Dict[str, Any]:
        """Build complete YAML configuration"""
        self.yaml_config = {
            'world': self._build_world_config(),
            'robot': self._build_robot_config(),
            'obstacle': self._build_obstacle_config()
        }
        return self.yaml_config

    def _build_world_config(self) -> Dict[str, Any]:
        """Build world configuration section"""
        world = self.map_builder.world
        return {
            'height': world.height,
            'width': world.width,
            'step_time': world.step_time,
            'sample_time': world.sample_time,
            'offset': world.offset,
            'collision_mode': world.collision_mode,
            'control_mode': world.control_mode
        }

    def _build_robot_config(self) -> Dict[str, Any]:
        """Build robot configuration section"""
        if not self.map_builder.robots:
            # Default robot configuration
            return {
                'kinematics': {'name': 'acker'},
                'shape': {'name': 'rectangle', 'length': 4.6, 'width': 1.6, 'wheelbase': 3},
                'state': [5, 5, 0, 0],
                'goal': [40, 40, 0],
                'vel_max': [4, 1],
                'behavior': {'name': 'dash'},
                'sensors': [
                    {
                        'type': 'lidar2d',
                        'range_min': 0,
                        'range_max': 20,
                        'angle_range': 3.14,
                        'number': 100,
                        'noise': False,
                        'std': 1,
                        'angle_std': 0.2,
                        'offset': [0, 0, 0],
                        'alpha': 0.4
                    }
                ]
            }

        # Use first robot's configuration (similar to ral_1220.yaml structure)
        robot = self.map_builder.robots[0]
        return {
            'kinematics': {'name': robot.kinematics},
            'shape': {'name': 'rectangle', 'length': 4.6, 'width': 1.6, 'wheelbase': 3},
            'state': robot.start_pos + [0],  # Add velocity component
            'goal': robot.goal_pos,
            'vel_max': robot.vel_max,
            'behavior': {'name': 'dash'},
            'sensors': [
                {
                    'type': 'lidar2d',
                    'range_min': 0,
                    'range_max': 20,
                    'angle_range': 3.14,
                    'number': 100,
                    'noise': False,
                    'std': 1,
                    'angle_std': 0.2,
                    'offset': [0, 0, 0],
                    'alpha': 0.4
                }
            ]
        }

    def _build_obstacle_config(self) -> List[Dict[str, Any]]:
        """Build obstacle configuration section"""
        if not self.map_builder.obstacles:
            return []

        # Group obstacles by material and shape for efficient organization
        obstacle_groups = self._group_obstacles()

        obstacle_config = []

        # Process each group
        for group_key, obstacles in obstacle_groups.items():
            shape_type, material = group_key

            if shape_type == ShapeType.POLYGON:
                # Handle polygon obstacles (similar to ral_1220.yaml structure)
                states = []
                shapes = []

                for obstacle in obstacles:
                    # Create state for obstacle center
                    states.append(obstacle.center + [0])  # Add angle component

                    # Create shape vertices (relative to center)
                    relative_vertices = []
                    for vertex in obstacle.vertices:
                        rel_x = vertex[0] - obstacle.center[0]
                        rel_y = vertex[1] - obstacle.center[1]
                        relative_vertices.append([rel_x, rel_y])
                    shapes.append(relative_vertices)

                obstacle_config.append({
                    'type': 'obstacle_polygon',
                    'number': len(obstacles),
                    'distribute': {
                        'mode': 'manual',
                        'states': states,
                        'shapes': shapes
                    }
                })

            elif shape_type == ShapeType.RECTANGLE:
                # Handle rectangle obstacles
                states = []
                shapes = []

                for obstacle in obstacles:
                    # Create state for obstacle center
                    states.append(obstacle.center + [obstacle.angle])

                    # Create rectangle shape (relative to center)
                    width, height = obstacle.dimensions
                    shapes.append([
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]
                    ])

                obstacle_config.append({
                    'type': 'obstacle_polygon',
                    'number': len(obstacles),
                    'distribute': {
                        'mode': 'manual',
                        'states': states,
                        'shapes': shapes
                    }
                })

            elif shape_type == ShapeType.CIRCLE:
                # Handle circle obstacles (approximated as polygons)
                states = []
                shapes = []

                for obstacle in obstacles:
                    # Create state for obstacle center
                    states.append(obstacle.center + [0])

                    # Create circle shape (relative to center)
                    relative_vertices = []
                    for vertex in obstacle.vertices:
                        rel_x = vertex[0] - obstacle.center[0]
                        rel_y = vertex[1] - obstacle.center[1]
                        relative_vertices.append([rel_x, rel_y])
                    shapes.append(relative_vertices)

                obstacle_config.append({
                    'type': 'obstacle_polygon',
                    'number': len(obstacles),
                    'distribute': {
                        'mode': 'manual',
                        'states': states,
                        'shapes': shapes
                    }
                })

        return obstacle_config

    def _group_obstacles(self) -> Dict[tuple, List[Obstacle]]:
        """Group obstacles by shape type and material"""
        groups = {}
        for obstacle in self.map_builder.obstacles:
            key = (obstacle.shape_type, obstacle.material)
            if key not in groups:
                groups[key] = []
            groups[key].append(obstacle)
        return groups

    def save_yaml(self, filename: str):
        """Save YAML configuration to file"""
        yaml_content = self.build_yaml()

        # Convert to YAML string format
        yaml_str = self._dict_to_yaml(yaml_content)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(yaml_str)

    def _dict_to_yaml(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to YAML string format"""
        yaml_lines = []
        indent_str = "  " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                yaml_lines.append(f"{indent_str}{key}:")
                yaml_lines.append(self._dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                yaml_lines.append(f"{indent_str}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        yaml_lines.append(f"{indent_str}  -")
                        yaml_lines.append(self._dict_to_yaml(item, indent + 2).rstrip())
                    else:
                        yaml_lines.append(f"{indent_str}  - {item}")
            elif isinstance(value, str):
                yaml_lines.append(f"{indent_str}{key}: '{value}'")
            else:
                yaml_lines.append(f"{indent_str}{key}: {value}")

        return "\n".join(yaml_lines)

class MapYAMLGenerator:
    """High-level interface for generating YAML from MapBuilder"""

    def __init__(self):
        pass

    @staticmethod
    def generate_yaml(map_builder: MapBuilder, filename: str):
        """Generate YAML configuration from MapBuilder"""
        # Validate map first
        issues = map_builder.validate_map()
        if issues:
            print("Warning: Map validation issues found:")
            for issue in issues:
                print(f"  - {issue}")

        # Build and save YAML
        builder = YAMLBuilder(map_builder)
        builder.save_yaml(filename)
        print(f"YAML configuration saved to: {filename}")

    @staticmethod
    def get_yaml_string(map_builder: MapBuilder) -> str:
        """Get YAML configuration as string"""
        builder = YAMLBuilder(map_builder)
        return builder._dict_to_yaml(builder.build_yaml())