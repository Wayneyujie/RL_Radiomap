"""
YAML to XML Converter for Sionna Electromagnetic Simulation
Converts IRSIM YAML configurations to Sionna XML scene descriptions
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from map_generator import MapBuilder, Obstacle, ShapeType, MaterialType
from resource_manager import ResourceManager

class YAMLToXMLConverter:
    """Converts YAML configurations to XML for Sionna electromagnetic simulation"""

    def __init__(self, mesh_directory: str = "meshes"):
        self.mesh_directory = mesh_directory
        self.resource_manager = ResourceManager()

    def convert_yaml_to_xml(self, yaml_file: str, xml_file: str,
                           generate_meshes: bool = True) -> bool:
        """Convert YAML configuration to XML scene description"""
        try:
            # Load YAML configuration
            yaml_config = self._load_yaml(yaml_file)

            # Parse obstacles from YAML
            obstacles = self._parse_obstacles_from_yaml(yaml_config)

            # Generate XML structure
            xml_root = self._generate_xml_structure(yaml_config, obstacles)

            # Save XML file
            self._save_xml(xml_root, xml_file)

            # Generate mesh files if requested
            if generate_meshes:
                self._generate_mesh_files(obstacles, self.mesh_directory)

            print(f"Successfully converted {yaml_file} to {xml_file}")
            return True

        except Exception as e:
            print(f"Error converting YAML to XML: {str(e)}")
            return False

    def convert_map_builder_to_xml(self, map_builder: MapBuilder, xml_file: str,
                                  mesh_directory: str = "meshes") -> bool:
        """Convert MapBuilder directly to XML"""
        try:
            self.mesh_directory = mesh_directory

            # Generate XML structure
            xml_root = self._generate_xml_structure_from_builder(map_builder)

            # Save XML file
            self._save_xml(xml_root, xml_file)

            # Generate mesh files
            self._generate_mesh_files(map_builder.obstacles, mesh_directory)

            print(f"Successfully generated XML from MapBuilder to {xml_file}")
            return True

        except Exception as e:
            print(f"Error converting MapBuilder to XML: {str(e)}")
            return False

    def _load_yaml(self, yaml_file: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        import yaml
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _parse_obstacles_from_yaml(self, yaml_config: Dict[str, Any]) -> List[Obstacle]:
        """Parse obstacles from YAML configuration"""
        obstacles = []

        if 'obstacle' not in yaml_config:
            return obstacles

        obstacle_configs = yaml_config['obstacle']
        if not isinstance(obstacle_configs, list):
            obstacle_configs = [obstacle_configs]

        for obs_config in obstacle_configs:
            if 'distribute' not in obs_config:
                continue

            distribute = obs_config['distribute']
            if 'states' not in distribute or 'shapes' not in distribute:
                continue

            states = distribute['states']
            shapes = distribute['shapes']

            for i, (state, shape) in enumerate(zip(states, shapes)):
                # Extract position and angle from state
                x, y, angle = state[0], state[1], state[2] if len(state) > 2 else 0.0

                # Calculate center from vertices
                vertices = shape
                if vertices and len(vertices) > 0:
                    center_x = x
                    center_y = y

                    # Determine shape type based on vertex pattern
                    if len(vertices) == 4 and self._is_rectangle(vertices):
                        shape_type = ShapeType.RECTANGLE
                        # Calculate dimensions
                        width = max(abs(v[0] - vertices[0][0]) for v in vertices)
                        height = max(abs(v[1] - vertices[0][1]) for v in vertices)
                        dimensions = [width, height]
                    elif len(vertices) > 8:
                        shape_type = ShapeType.CIRCLE
                        # Estimate radius
                        max_dist = max(np.sqrt((v[0])**2 + (v[1])**2) for v in vertices)
                        dimensions = [max_dist]
                    else:
                        shape_type = ShapeType.POLYGON
                        dimensions = [0, 0]

                    obstacle = Obstacle(
                        shape_type=shape_type,
                        vertices=vertices,
                        center=[center_x, center_y],
                        dimensions=dimensions,
                        material=MaterialType.CONCRETE,  # Default material
                        angle=angle
                    )
                    obstacles.append(obstacle)

        return obstacles

    def _is_rectangle(self, vertices: List[List[float]]) -> bool:
        """Check if vertices form a rectangle"""
        if len(vertices) != 4:
            return False

        # Simple heuristic: check for axis-aligned rectangle
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        # Check if there are only 2 unique x and y values
        return len(set(xs)) == 2 and len(set(ys)) == 2

    def _generate_xml_structure(self, yaml_config: Dict[str, Any], obstacles: List[Obstacle]) -> ET.Element:
        """Generate XML structure from YAML config and obstacles"""
        # Create root scene element
        scene = ET.Element('scene', version="2.1.0")

        # Add integrator
        integrator = ET.SubElement(scene, 'integrator', type="path", id="elm__0", name="elm__0")
        ET.SubElement(integrator, 'integer', name="max_depth", value="12")

        # Add materials
        materials = self.resource_manager.generate_material_definitions()
        for material in materials:
            mat_elem = ET.SubElement(scene, material['type'], id=material['id'], name=material['name'])
            for attr_name, attr_value in material['attributes'].items():
                attr_elem = ET.SubElement(mat_elem, attr_value['type'], name=attr_name)
                attr_elem.set('value', attr_value['value'])

        # Add shapes (obstacles)
        shapes = self.resource_manager.generate_shape_definitions(obstacles, self.mesh_directory)
        for shape in shapes:
            if shape['type'] == 'ply':
                shape_elem = ET.SubElement(scene, 'ply', id=shape['id'], name=shape['name'])
                ET.SubElement(shape_elem, 'string', name="filename", value=shape['filename'])
                ET.SubElement(shape_elem, 'ref', id=shape['material_ref'], name="bsdf")
            else:
                # Dynamic shape (cube)
                shape_elem = ET.SubElement(scene, 'cube', id=shape['id'], name=shape['name'])

                # Add transform
                transform_elem = ET.SubElement(shape_elem, 'transform', name="to_world")

                # Scale
                scale_elem = ET.SubElement(transform_elem, 'scale')
                scale_elem.set('x', str(shape['transform']['scale']['x']))
                scale_elem.set('y', str(shape['transform']['scale']['y']))
                scale_elem.set('z', str(shape['transform']['scale']['z']))

                # Translate
                translate_elem = ET.SubElement(transform_elem, 'translate')
                translate_elem.set('x', str(shape['transform']['translate']['x']))
                translate_elem.set('y', str(shape['transform']['translate']['y']))
                translate_elem.set('z', str(shape['transform']['translate']['z']))

                # Rotate
                if 'rotate' in shape['transform']:
                    rotate_elem = ET.SubElement(transform_elem, 'rotate')
                    rotate_elem.set('axis', shape['transform']['rotate']['axis'])
                    rotate_elem.set('angle', str(shape['transform']['rotate']['angle']))

                ET.SubElement(shape_elem, 'ref', id=shape['material_ref'])

        return scene

    def _generate_xml_structure_from_builder(self, map_builder: MapBuilder) -> ET.Element:
        """Generate XML structure directly from MapBuilder"""
        # Create root scene element
        scene = ET.Element('scene', version="2.1.0")

        # Add integrator
        integrator = ET.SubElement(scene, 'integrator', type="path", id="elm__0", name="elm__0")
        ET.SubElement(integrator, 'integer', name="max_depth", value="12")

        # Add materials
        materials = self.resource_manager.generate_material_definitions()
        for material in materials:
            mat_elem = ET.SubElement(scene, material['type'], id=material['id'], name=material['name'])
            for attr_name, attr_value in material['attributes'].items():
                attr_elem = ET.SubElement(mat_elem, attr_value['type'], name=attr_name)
                attr_elem.set('value', attr_value['value'])

        # Add shapes (obstacles)
        shapes = self.resource_manager.generate_shape_definitions(map_builder.obstacles, self.mesh_directory)
        for shape in shapes:
            if shape['type'] == 'ply':
                shape_elem = ET.SubElement(scene, 'ply', id=shape['id'], name=shape['name'])
                ET.SubElement(shape_elem, 'string', name="filename", value=shape['filename'])
                ET.SubElement(shape_elem, 'ref', id=shape['material_ref'], name="bsdf")
            else:
                # Dynamic shape
                shape_elem = ET.SubElement(scene, 'cube', id=shape['id'], name=shape['name'])

                # Add transform
                transform_elem = ET.SubElement(shape_elem, 'transform', name="to_world")

                # Scale
                scale_elem = ET.SubElement(transform_elem, 'scale')
                scale_elem.set('x', str(shape['transform']['scale']['x']))
                scale_elem.set('y', str(shape['transform']['scale']['y']))
                scale_elem.set('z', str(shape['transform']['scale']['z']))

                # Translate
                translate_elem = ET.SubElement(transform_elem, 'translate')
                translate_elem.set('x', str(shape['transform']['translate']['x']))
                translate_elem.set('y', str(shape['transform']['translate']['y']))
                translate_elem.set('z', str(shape['transform']['translate']['z']))

                # Rotate
                if 'rotate' in shape['transform']:
                    rotate_elem = ET.SubElement(transform_elem, 'rotate')
                    rotate_elem.set('axis', shape['transform']['rotate']['axis'])
                    rotate_elem.set('angle', str(shape['transform']['rotate']['angle']))

                ET.SubElement(shape_elem, 'ref', id=shape['material_ref'])

        return scene

    def _save_xml(self, xml_root: ET.Element, xml_file: str):
        """Save XML tree to file with proper formatting"""
        # Convert to string with proper formatting
        rough_string = ET.tostring(xml_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")

        # Remove empty lines
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])

        # Save to file
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

    def _generate_mesh_files(self, obstacles: List[Obstacle], mesh_directory: str):
        """Generate PLY mesh files for obstacles"""
        os.makedirs(mesh_directory, exist_ok=True)

        for i, obstacle in enumerate(obstacles):
            mesh_name = self.resource_manager.create_dynamic_mesh_name(
                obstacle.material, obstacle.shape_type, i
            )

            if not self.resource_manager._is_predefined_mesh(mesh_name):
                # Generate PLY file
                self._create_ply_file(obstacle, os.path.join(mesh_directory, f"{mesh_name}.ply"))

    def _create_ply_file(self, obstacle: Obstacle, ply_file: str):
        """Create a PLY file for the obstacle"""
        # Convert 2D vertices to 3D with default height
        height = 3.0

        # Generate vertices for 3D mesh (extrude 2D shape)
        vertices = []
        faces = []

        # Bottom vertices
        for vertex in obstacle.vertices:
            vertices.append([vertex[0], vertex[1], 0])

        # Top vertices
        for vertex in obstacle.vertices:
            vertices.append([vertex[0], vertex[1], height])

        # Generate faces
        n_vertices_2d = len(obstacle.vertices)

        # Bottom face
        bottom_face = list(range(n_vertices_2d))
        faces.append(bottom_face)

        # Top face
        top_face = list(range(n_vertices_2d, 2 * n_vertices_2d))
        faces.append(top_face[::-1])  # Reverse for correct winding

        # Side faces
        for i in range(n_vertices_2d):
            next_i = (i + 1) % n_vertices_2d
            # Quad face (two triangles)
            faces.append([i, next_i, next_i + n_vertices_2d])
            faces.append([i, next_i + n_vertices_2d, i + n_vertices_2d])

        # Write PLY file
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces
            for face in faces:
                f.write(f"{len(face)} {' '.join(map(str, face))}\n")

class XMLConverter:
    """High-level interface for XML conversion"""

    def __init__(self):
        pass

    @staticmethod
    def convert_yaml_file(yaml_file: str, xml_file: str, mesh_directory: str = "meshes") -> bool:
        """Convert YAML file to XML"""
        converter = YAMLToXMLConverter(mesh_directory)
        return converter.convert_yaml_to_xml(yaml_file, xml_file)

    @staticmethod
    def convert_map_builder(map_builder: MapBuilder, xml_file: str, mesh_directory: str = "meshes") -> bool:
        """Convert MapBuilder to XML"""
        converter = YAMLToXMLConverter(mesh_directory)
        return converter.convert_map_builder_to_xml(map_builder, xml_file, mesh_directory)