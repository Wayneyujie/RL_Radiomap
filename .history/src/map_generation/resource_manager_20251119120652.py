"""
Resource Manager for Materials and Mesh Files
Handles material properties and mesh file assignments for XML generation
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
from .map_generator import MaterialType, ShapeType

class MaterialProperties:
    """Material properties for electromagnetic simulation"""

    def __init__(self, name: str, reflectance: Tuple[float, float, float],
                 mesh_prefix: str = "mesh"):
        self.name = name
        self.reflectance = reflectance
        self.mesh_prefix = mesh_prefix

class ResourceManager:
    """Manages materials and mesh resources for XML generation"""

    def __init__(self):
        self.materials = self._initialize_materials()
        self.mesh_library = self._initialize_mesh_library()
        self.mesh_counter = {}

    def _initialize_materials(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize material properties based on INVS.xml"""
        return {
            MaterialType.MARBLE: MaterialProperties(
                name="mat-itu_marble",
                reflectance=(1.0, 0.0, 0.3),
                mesh_prefix="Wall"
            ),
            MaterialType.GLASS: MaterialProperties(
                name="mat-itu_glass",
                reflectance=(1.0, 1.0, 1.0),
                mesh_prefix="Glass"
            ),
            MaterialType.METAL: MaterialProperties(
                name="mat-itu_metal",
                reflectance=(1.0, 0.0, 0.3),
                mesh_prefix="Metal"
            ),
            MaterialType.CONCRETE: MaterialProperties(
                name="mat-itu_concrete",
                reflectance=(1.0, 0.0, 0.3),
                mesh_prefix="Wall"
            )
        }

    def _initialize_mesh_library(self) -> Dict[str, List[str]]:
        """Initialize mesh file library based on existing INVS meshes"""
        # Based on existing mesh files in radio/INVS2/meshes/
        return {
            "Wall": [f"Wall{i:02d}" for i in range(1, 25)],  # Wall01 to Wall24
            "Glass": [f"Glass{i:02d}" for i in range(1, 5)],   # Glass01 to Glass04
            "Metal": [f"Metal{i:02d}" for i in range(1, 5)],   # Metal01 to Metal04
            "Mesh": [f"Mesh{i:02d}" for i in range(90, 102)],  # Mesh90 to Mesh101
            "Cube": ["my_added_wall"]  # For dynamic obstacles
        }

    def get_material_properties(self, material: MaterialType) -> MaterialProperties:
        """Get material properties for given material type"""
        return self.materials.get(material, self.materials[MaterialType.CONCRETE])

    def assign_mesh_file(self, material: MaterialType, shape_type: ShapeType,
                        obstacle_index: int) -> str:
        """Assign mesh file to obstacle based on material and shape type"""
        material_props = self.get_material_properties(material)

        # Create mesh key for counter
        mesh_key = f"{material_props.mesh_prefix}_{shape_type.value}"

        if mesh_key not in self.mesh_counter:
            self.mesh_counter[mesh_key] = 0

        self.mesh_counter[mesh_key] += 1
        counter = self.mesh_counter[mesh_key]

        # Select appropriate mesh library
        if shape_type == ShapeType.RECTANGLE and "Wall" in self.mesh_library:
            mesh_list = self.mesh_library["Wall"]
        elif shape_type == ShapeType.POLYGON:
            mesh_list = self.mesh_library.get("Mesh", self.mesh_library["Wall"])
        else:
            mesh_list = self.mesh_library["Wall"]

        # Cycle through available meshes
        mesh_index = (counter - 1) % len(mesh_list)
        mesh_name = mesh_list[mesh_index]

        return mesh_name

    def create_dynamic_mesh_name(self, material: MaterialType, shape_type: ShapeType,
                                obstacle_index: int) -> str:
        """Create dynamic mesh name for new obstacles"""
        material_props = self.get_material_properties(material)

        if shape_type == ShapeType.CIRCLE:
            return f"{material_props.mesh_prefix}_circle_{obstacle_index:03d}"
        elif shape_type == ShapeType.RECTANGLE:
            return f"{material_props.mesh_prefix}_rect_{obstacle_index:03d}"
        else:
            return f"{material_props.mesh_prefix}_poly_{obstacle_index:03d}"

    def generate_material_definitions(self) -> List[Dict]:
        """Generate material definitions for XML"""
        material_defs = []

        for material_type, material_props in self.materials.items():
            material_def = {
                'type': 'bsdf',
                'id': material_props.name,
                'name': material_props.name,
                'attributes': {
                    'reflectance': {
                        'type': 'rgb',
                        'value': f"{material_props.reflectance[0]:.6f} "
                               f"{material_props.reflectance[1]:.6f} "
                               f"{material_props.reflectance[2]:.6f}"
                    }
                }
            }
            material_defs.append(material_def)

        return material_defs

    def generate_shape_definitions(self, obstacles: List, mesh_directory: str = "meshes") -> List[Dict]:
        """Generate shape definitions for XML"""
        shape_defs = []

        for i, obstacle in enumerate(obstacles):
            material_props = self.get_material_properties(obstacle.material)
            mesh_name = self.assign_mesh_file(obstacle.material, obstacle.shape_type, i)

            # Check if it's a predefined mesh or needs dynamic generation
            if self._is_predefined_mesh(mesh_name):
                shape_def = {
                    'type': 'ply',
                    'id': f"mesh-{mesh_name}",
                    'name': f"mesh-{mesh_name}",
                    'filename': os.path.join(mesh_directory, f"{mesh_name}.ply"),
                    'material_ref': material_props.name
                }
            else:
                # Dynamic shape generation (for custom obstacles)
                shape_def = {
                    'type': 'cube',
                    'id': mesh_name,
                    'name': mesh_name,
                    'transform': self._generate_transform(obstacle),
                    'material_ref': material_props.name
                }

            shape_defs.append(shape_def)

        return shape_defs

    def _is_predefined_mesh(self, mesh_name: str) -> bool:
        """Check if mesh is predefined in the library"""
        for mesh_list in self.mesh_library.values():
            if mesh_name in mesh_list:
                return True
        return False

    def _generate_transform(self, obstacle) -> Dict:
        """Generate transform for dynamic obstacles"""
        # Calculate dimensions for transform
        if obstacle.shape_type == ShapeType.RECTANGLE:
            width, length = obstacle.dimensions
            height = 3.0  # Default height
        elif obstacle.shape_type == ShapeType.CIRCLE:
            radius = obstacle.dimensions[0]
            width = length = radius * 2
            height = 3.0
        else:
            # Default for polygons
            width = length = 2.0
            height = 3.0

        return {
            'scale': {
                'x': width,
                'y': length,
                'z': height
            },
            'translate': {
                'x': obstacle.center[0],
                'y': obstacle.center[1],
                'z': height / 2  # Center vertically
            },
            'rotate': {
                'axis': 'z',
                'angle': obstacle.angle
            }
        }

    def export_material_config(self, filename: str):
        """Export material configuration to JSON file"""
        config = {
            'materials': {},
            'mesh_library': self.mesh_library
        }

        for material_type, material_props in self.materials.items():
            config['materials'][material_type.value] = {
                'name': material_props.name,
                'reflectance': material_props.reflectance,
                'mesh_prefix': material_props.mesh_prefix
            }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    def get_statistics(self) -> Dict:
        """Get resource usage statistics"""
        return {
            'total_materials': len(self.materials),
            'mesh_assignments': dict(self.mesh_counter),
            'available_meshes': {
                category: len(meshes) for category, meshes in self.mesh_library.items()
            }
        }