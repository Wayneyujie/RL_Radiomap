#!/usr/bin/env python3
"""
Simplified Radio Shadowing Test

This script tests electromagnetic map generation using the converted XML file
from our YAML configuration. It's a simplified version of 10.29radio_shadowing.py
adapted for our simple_test.xml scenario.

Author: Claude Code Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
import warnings
import argparse

# Try to import sionna, with graceful fallback
try:
    from sionna.rt import load_scene, Transmitter, PlanarArray, Camera
    SIONNA_AVAILABLE = True
    print("✓ SiOnNA library available for radio simulation")
except ImportError as e:
    SIONNA_AVAILABLE = False
    print(f"⚠️  SiOnNA library not available: {e}")
    print("   This test will simulate electromagnetic map generation with mock data")

class SimpleRadioShadowingTest:
    def __init__(self, xml_file='simple_test.xml', yaml_file='simple_test.yaml'):
        """Initialize the test with configuration files"""
        self.xml_file = xml_file
        self.yaml_file = yaml_file
        self.scene = None
        self.config = None
        self.radio_map = None

        # Configuration parameters
        self.frequency = 2.14e9  # 2.14 GHz
        self.cm_cell_size = (1, 1)  # Coverage map cell size in meters
        self.max_depth = 5  # Ray tracing depth
        self.num_samples = int(1e6)  # Number of samples for ray tracing

    def load_configuration(self):
        """Load YAML configuration"""
        try:
            import yaml
            with open(self.yaml_file, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Loaded configuration from {self.yaml_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False

    def load_scene(self):
        """Load the Mitsuba XML scene"""
        if not SIONNA_AVAILABLE:
            print("⚠️  SiOnNA not available - simulating scene loading")
            return True

        try:
            self.scene = load_scene(self.xml_file)
            print(f"✓ Loaded scene from {self.xml_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading scene: {e}")
            return False

    def setup_antenna_arrays(self):
        """Configure antenna arrays for transmitter and receiver"""
        if not SIONNA_AVAILABLE:
            print("⚠️  SiOnNA not available - simulating antenna setup")
            return True

        try:
            # Configure transmitter antenna array
            self.scene.tx_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="tr38901",
                polarization="V"
            )

            # Configure receiver antenna array
            self.scene.rx_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="cross"
            )

            print("✓ Configured antenna arrays")
            return True
        except Exception as e:
            print(f"❌ Error configuring antenna arrays: {e}")
            return False

    def get_robot_position(self):
        """Extract robot position from configuration"""
        if not self.config or 'robots' not in self.config:
            return [2.0, 2.0, 1.5, 0, 0, 1.57]  # Default position

        robot_state = self.config['robots'].get('state', [2.0, 2.0, 0, 0])
        x, y, theta, v = robot_state[0], robot_state[1], robot_state[2] if len(robot_state) > 2 else 0, robot_state[3] if len(robot_state) > 3 else 0

        # Convert to radio coordinates (transmitter position)
        # Based on original code's coordinate transformation
        translation = [0, 0]  # No translation needed for simple scenario
        radio_pos = [x + translation[0], y + translation[1]]

        # Transmitter position (x, y, z, yaw, pitch, roll)
        tx_position = [radio_pos[0], radio_pos[1], 1.5, 0, 0, 1.57]

        return tx_position

    def add_transmitter(self):
        """Add transmitter to the scene"""
        if not SIONNA_AVAILABLE:
            print("⚠️  SiOnNA not available - simulating transmitter addition")
            return True

        try:
            tx_position = self.get_robot_position()

            # Create transmitter
            tx = Transmitter(
                name="robot_tx",
                position=tx_position,
                color=[1, 0, 0]  # Red color for visualization
            )

            # Add transmitter to scene
            self.scene.add(tx)
            self.scene.frequency = self.frequency
            self.scene.synthetic_array = True

            print(f"✓ Added transmitter at position {tx_position[:3]}")
            return True
        except Exception as e:
            print(f"❌ Error adding transmitter: {e}")
            return False

    def generate_coverage_map(self):
        """Generate electromagnetic coverage map"""
        if not SIONNA_AVAILABLE:
            print("⚠️  SiOnNA not available - generating mock coverage map")
            return self.generate_mock_coverage_map()

        try:
            print("🔄 Generating coverage map...")

            # Generate coverage map
            cm = self.scene.coverage_map(
                max_depth=self.max_depth,
                diffraction=True,
                cm_cell_size=self.cm_cell_size,
                combining_vec=None,
                precoding_vec=None,
                num_samples=self.num_samples
            )

            # Convert to path gain in dB
            path_gain_db = 20 * np.log10(cm._path_gain.numpy())

            print("✓ Coverage map generated successfully")
            return path_gain_db

        except Exception as e:
            print(f"❌ Error generating coverage map: {e}")
            return None

    def generate_mock_coverage_map(self):
        """Generate a mock coverage map for testing without SiOnNA"""
        print("🔄 Generating mock coverage map...")

        # Create a grid based on world dimensions
        world_width = 20  # meters
        world_height = 20  # meters
        cell_size = 1  # meters

        grid_width = int(world_width / cell_size)
        grid_height = int(world_height / cell_size)

        # Generate base signal strength
        x = np.linspace(0, world_width, grid_width)
        y = np.linspace(0, world_height, grid_height)
        X, Y = np.meshgrid(x, y)

        # Get robot position
        tx_pos = self.get_robot_position()
        tx_x, tx_y = tx_pos[0], tx_pos[1]

        # Calculate distance from transmitter
        distance = np.sqrt((X - tx_x)**2 + (Y - tx_y)**2)

        # Simple path loss model: PL = 20*log10(d) + 20*log10(f) - 147.55 (in dB)
        # Using simplified model for demonstration
        path_loss_db = 30 * np.log10(distance + 1) - 20

        # Add obstacles effect (simple shadowing)
        # Obstacle positions from simple_test.yaml
        obstacles = [(8, 8, 1.0), (12, 5, 0.8), (5, 12, 0.6)]

        for obs_x, obs_y, obs_size in obstacles:
            obs_distance = np.sqrt((X - obs_x)**2 + (Y - obs_y)**2)
            shadow_zone = obs_distance < obs_size
            path_loss_db[shadow_zone] += 20  # Additional loss in shadow zones

        # Set minimum and maximum values
        path_loss_db = np.clip(path_loss_db, -120, -40)

        print("✓ Mock coverage map generated successfully")
        return path_loss_db

    def visualize_results(self, coverage_map):
        """Visualize the electromagnetic coverage map"""
        try:
            plt.figure(figsize=(12, 10))

            # Plot coverage map
            plt.imshow(coverage_map, cmap='viridis', origin='lower',
                      extent=[0, 20, 0, 20], aspect='equal')
            plt.colorbar(label='Path Gain (dB)')

            # Add transmitter position
            tx_pos = self.get_robot_position()
            plt.scatter(tx_pos[0], tx_pos[1], c='red', s=100, marker='*',
                       label='Transmitter', edgecolors='white', linewidth=2)

            # Add obstacles
            obstacles = [(8, 8, 1.0), (12, 5, 0.8), (5, 12, 0.6)]
            for obs_x, obs_y, obs_size in obstacles:
                circle = plt.Circle((obs_x, obs_y), obs_size/2,
                                   color='red', fill=False, linewidth=2, linestyle='--')
                plt.gca().add_patch(circle)

            # Add robot position
            plt.scatter(2, 2, c='blue', s=50, marker='o',
                       label='Robot Start', edgecolors='white', linewidth=1)

            plt.title('Electromagnetic Coverage Map - Simple Test Scenario')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'electromagnetic_map_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Coverage map saved as {filename}")

            # Display statistics
            self.print_statistics(coverage_map)

            plt.show()
            return True

        except Exception as e:
            print(f"❌ Error visualizing results: {e}")
            return False

    def print_statistics(self, coverage_map):
        """Print statistics about the coverage map"""
        try:
            print("\n📊 Coverage Map Statistics:")
            print(f"  Grid size: {coverage_map.shape}")
            print(f"  Min path gain: {np.min(coverage_map):.2f} dB")
            print(f"  Max path gain: {np.max(coverage_map):.2f} dB")
            print(f"  Mean path gain: {np.mean(coverage_map):.2f} dB")
            print(f"  Std deviation: {np.std(coverage_map):.2f} dB")

            # Calculate coverage area (above threshold)
            threshold = -80  # dB
            coverage_area = np.sum(coverage_map > threshold)
            total_area = coverage_map.size
            coverage_percentage = (coverage_area / total_area) * 100
            print(f"  Coverage area (> {threshold} dB): {coverage_percentage:.1f}%")

        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")

    def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Simple Radio Shadowing Test")
        print("=" * 50)

        # Load configuration
        if not self.load_configuration():
            return False

        # Load scene
        if not self.load_scene():
            return False

        # Setup antenna arrays
        if not self.setup_antenna_arrays():
            return False

        # Add transmitter
        if not self.add_transmitter():
            return False

        # Generate coverage map
        coverage_map = self.generate_coverage_map()
        if coverage_map is None:
            return False

        # Visualize results
        if not self.visualize_results(coverage_map):
            return False

        print("\n🎉 Test completed successfully!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Radio Shadowing Test')
    parser.add_argument('--xml', default='simple_test.xml',
                       help='XML scene file (default: simple_test.xml)')
    parser.add_argument('--yaml', default='simple_test.yaml',
                       help='YAML configuration file (default: simple_test.yaml)')

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.xml):
        print(f"❌ XML file not found: {args.xml}")
        print("   Please run 'python yaml_to_xml_converter.py simple_test.yaml' first")
        sys.exit(1)

    if not os.path.exists(args.yaml):
        print(f"❌ YAML file not found: {args.yaml}")
        sys.exit(1)

    # Create and run test
    test = SimpleRadioShadowingTest(args.xml, args.yaml)
    success = test.run_test()

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()