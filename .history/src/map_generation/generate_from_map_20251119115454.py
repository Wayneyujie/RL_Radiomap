"""
Main Interface for Map Generation Pipeline
Complete workflow: Python API → YAML → XML → Sionna Electromagnetic Maps
"""

import os
import sys
from typing import Optional
from map_generator import MapBuilder, MaterialType
from yaml_builder import MapYAMLGenerator
from yaml_to_xml_converter import XMLConverter

class MapGenerationPipeline:
    """Complete pipeline for map generation and conversion"""

    def __init__(self, output_directory: str = "generated_maps"):
        self.output_directory = output_directory
        self.map_builder = MapBuilder()
        os.makedirs(output_directory, exist_ok=True)

    def create_map(self) -> MapBuilder:
        """Create a new map and return the builder for configuration"""
        self.map_builder = MapBuilder()
        return self.map_builder

    def generate_files(self, base_name: str,
                      generate_yaml: bool = True,
                      generate_xml: bool = True,
                      mesh_directory: str = "meshes") -> tuple:
        """Generate YAML and XML files from the current map configuration"""

        # Create full paths
        yaml_path = os.path.join(self.output_directory, f"{base_name}.yaml")
        xml_path = os.path.join(self.output_directory, f"{base_name}.xml")
        mesh_path = os.path.join(self.output_directory, mesh_directory)

        yaml_success = True
        xml_success = True

        # Generate YAML file
        if generate_yaml:
            try:
                MapYAMLGenerator.generate_yaml(self.map_builder, yaml_path)
            except Exception as e:
                print(f"Error generating YAML: {e}")
                yaml_success = False

        # Generate XML file
        if generate_xml:
            try:
                XMLConverter.convert_map_builder(self.map_builder, xml_path, mesh_path)
            except Exception as e:
                print(f"Error generating XML: {e}")
                xml_success = False

        return yaml_success, xml_success

    def generate_and_run_simulation(self, base_name: str, sim_main_path: Optional[str] = None):
        """Generate files and optionally run sim_main_sh_fixed.py"""
        yaml_success, xml_success = self.generate_files(base_name)

        if not yaml_success:
            print("Failed to generate YAML file. Cannot run simulation.")
            return False

        if not xml_success:
            print("Failed to generate XML file. Electromagnetic mapping may not work.")

        # Run simulation if sim_main_path is provided
        if sim_main_path:
            yaml_path = os.path.join(self.output_directory, f"{base_name}.yaml")
            cmd = f"python {sim_main_path} --config {yaml_path}"
            print(f"Running simulation: {cmd}")
            os.system(cmd)

        return yaml_success and xml_success

def create_example_map_basic():
    """Create a basic example map"""
    pipeline = MapGenerationPipeline("examples")

    # Configure map
    builder = pipeline.create_map()
    builder.set_world_size(22, 22)
    builder.add_robot([5, 5, 0], [18, 18, 0])

    # Add some walls
    builder.add_rectangle_wall([2, 2], [20, 2], MaterialType.CONCRETE)  # Bottom wall
    builder.add_rectangle_wall([2, 2], [2, 20], MaterialType.CONCRETE)  # Left wall
    builder.add_rectangle_wall([20, 2], [20, 20], MaterialType.CONCRETE)  # Right wall
    builder.add_rectangle_wall([2, 20], [20, 20], MaterialType.CONCRETE)  # Top wall

    # Add some obstacles
    builder.add_circle_obstacle([10, 10], 1.5, MaterialType.MARBLE)
    builder.add_rectangle_wall([8, 15], [12, 15.2], MaterialType.GLASS, width=0.2)

    # Generate files
    yaml_success, xml_success = pipeline.generate_files("basic_example")

    print(f"Basic example map generated:")
    print(f"  YAML: {'✓' if yaml_success else '✗'}")
    print(f"  XML:  {'✓' if xml_success else '✗'}")
    print(f"  Location: {pipeline.output_directory}")

def create_example_map_complex():
    """Create a complex example map with various obstacle types"""
    pipeline = MapGenerationPipeline("examples")

    # Configure map
    builder = pipeline.create_map()
    builder.set_world_size(30, 30)

    # Add multiple robots
    builder.add_robot([5, 5, 0], [25, 25, 0])
    builder.add_robot([25, 5, np.pi/2], [5, 25, -np.pi/2])

    # Create maze-like structure
    # Outer walls
    builder.add_rectangle_wall([1, 1], [29, 1], MaterialType.CONCRETE)
    builder.add_rectangle_wall([1, 1], [1, 29], MaterialType.CONCRETE)
    builder.add_rectangle_wall([29, 1], [29, 29], MaterialType.CONCRETE)
    builder.add_rectangle_wall([1, 29], [29, 29], MaterialType.CONCRETE)

    # Internal walls
    builder.add_rectangle_wall([10, 1], [10, 15], MaterialType.MARBLE)
    builder.add_rectangle_wall([20, 15], [20, 29], MaterialType.METAL)
    builder.add_rectangle_wall([1, 20], [15, 20], MaterialType.GLASS, width=0.3)
    builder.add_rectangle_wall([15, 10], [29, 10], MaterialType.CONCRETE, width=0.2)

    # Add random obstacles
    builder.add_random_circles(5, (0.8, 2.0), (3, 3, 12, 12), MaterialType.MARBLE)
    builder.add_random_circles(3, (1.0, 1.8), (18, 18, 27, 27), MaterialType.METAL)
    builder.add_random_rectangles(4, (1.5, 3.0, 1.5, 3.0), (12, 3, 18, 8), MaterialType.GLASS)

    # Add polygon obstacles
    builder.add_polygon_obstacle([
        [7, 22], [9, 24], [8, 26], [6, 25], [5, 23]
    ], MaterialType.CONCRETE)

    builder.add_polygon_obstacle([
        [22, 7], [24, 8], [25, 10], [23, 11], [21, 9]
    ], MaterialType.MARBLE)

    # Generate files
    yaml_success, xml_success = pipeline.generate_files("complex_example")

    print(f"Complex example map generated:")
    print(f"  YAML: {'✓' if yaml_success else '✗'}")
    print(f"  XML:  {'✓' if xml_success else '✗'}")
    print(f"  Location: {pipeline.output_directory}")
    print(f"  Obstacles: {builder.get_obstacle_count()}")
    print(f"  Robots: {builder.get_robot_count()}")

def convert_existing_yaml(yaml_file: str, output_name: Optional[str] = None):
    """Convert existing YAML file to XML"""
    if not os.path.exists(yaml_file):
        print(f"YAML file not found: {yaml_file}")
        return False

    if output_name is None:
        base_name = os.path.splitext(os.path.basename(yaml_file))[0]
        output_dir = os.path.dirname(yaml_file)
    else:
        base_name = output_name
        output_dir = "generated_maps"

    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, f"{base_name}.xml")

    success = XMLConverter.convert_yaml_file(yaml_file, xml_path)

    if success:
        print(f"Successfully converted {yaml_file} to {xml_path}")
    else:
        print(f"Failed to convert {yaml_file}")

    return success

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generate_from_map.py <command> [options]")
        print("")
        print("Commands:")
        print("  basic              - Generate basic example map")
        print("  complex            - Generate complex example map")
        print("  convert <yaml>     - Convert existing YAML to XML")
        print("  interactive        - Start interactive map builder")
        return

    command = sys.argv[1].lower()

    if command == "basic":
        create_example_map_basic()

    elif command == "complex":
        create_example_map_complex()

    elif command == "convert":
        if len(sys.argv) < 3:
            print("Usage: python generate_from_map.py convert <yaml_file> [output_name]")
            return
        yaml_file = sys.argv[2]
        output_name = sys.argv[3] if len(sys.argv) > 3 else None
        convert_existing_yaml(yaml_file, output_name)

    elif command == "interactive":
        print("Interactive map builder - implement your custom logic here")
        # This could be expanded to include a CLI or GUI interface
        builder = MapBuilder()
        # Users would configure the builder programmatically here
        print("Use the MapBuilder API directly in your Python code")

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    # Import numpy for the examples
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available. Some features may not work.")
        # Define a simple replacement for numpy functions
        class np:
            @staticmethod
            def pi():
                return 3.141592653589793
            @staticmethod
            def arctan2(y, x):
                import math
                return math.atan2(y, x)
            @staticmethod
            def sqrt(x):
                import math
                return math.sqrt(x)
            @staticmethod
            def cos(x):
                import math
                return math.cos(x)
            @staticmethod
            def sin(x):
                import math
                return math.sin(x)
            @staticmethod
            def uniform(a, b):
                import random
                return random.uniform(a, b)
            @staticmethod
            def mean(arr):
                import math
                return sum(arr) / len(arr)

    main()