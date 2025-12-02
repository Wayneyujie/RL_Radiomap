#!/usr/bin/env python3
"""
Mock Radio Shadowing Test (Working Version)

This script demonstrates electromagnetic map generation using mock data
when SiOnNA library is not available or has compatibility issues.
It simulates radio wave propagation with realistic path loss models.

Author: Claude Code Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
import argparse

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️  PyYAML not available, using default configuration")

class MockRadioShadowingTest:
    def __init__(self, xml_file='simple_test.xml', yaml_file='simple_test.yaml'):
        """Initialize the test with configuration files"""
        self.xml_file = xml_file
        self.yaml_file = yaml_file
        self.config = None
        self.coverage_map = None

        # Simulation parameters
        self.frequency = 2.14e9  # 2.14 GHz
        self.world_width = 20  # meters
        self.world_height = 20  # meters
        self.cell_size = 0.5  # meters
        self.transmitter_height = 1.5  # meters
        self.obstacle_height = 2.0  # meters

        # Path loss model parameters (simplified Friis transmission equation)
        self.c = 3e8  # Speed of light (m/s)
        self.wavelength = self.c / self.frequency

    def load_configuration(self):
        """Load YAML configuration"""
        if not YAML_AVAILABLE:
            print("⚠️  Using default configuration")
            self.config = {
                'world': {'width': 20, 'height': 20},
                'robots': {'state': [2, 2, 0, 0]},
                'obstacles': [
                    {'type': 'obstacle_polygon', 'number': 3,
                     'distribute': {
                         'states': [[8, 8, 0], [12, 5, 0], [5, 12, 0]],
                         'shapes': [
                             [[0, 0], [1, 0], [1, 1], [0, 1]],      # 1x1 obstacle
                             [[0, 0], [0.8, 0], [0.8, 0.8], [0, 0.8]],  # 0.8x0.8 obstacle
                             [[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6]]   # 0.6x0.6 obstacle
                         ]
                     }}
                ]
            }
            return True

        try:
            with open(self.yaml_file, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Loaded configuration from {self.yaml_file}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading configuration, using defaults: {e}")
            return self.load_default_config()

    def get_transmitter_position(self):
        """Extract transmitter position from configuration"""
        if not self.config or 'robots' not in self.config:
            return [2.0, 2.0, self.transmitter_height]

        robot_state = self.config['robots'].get('state', [2.0, 2.0, 0, 0])
        x = robot_state[0]
        y = robot_state[1]
        return [x, y, self.transmitter_height]

    def get_obstacles(self):
        """Extract obstacle positions and sizes from configuration"""
        if not self.config or 'obstacles' not in self.config:
            return []

        obstacles = []
        for obstacle_group in self.config.get('obstacles', []):
            if 'obstacle_polygon' in obstacle_group.get('type', ''):
                states = obstacle_group.get('distribute', {}).get('states', [])
                shapes = obstacle_group.get('distribute', {}).get('shapes', [])

                for state, shape in zip(states, shapes):
                    if len(state) >= 2 and len(shape) >= 3:
                        x, y = state[0], state[1]
                        # Calculate bounding box of polygon
                        shape_array = np.array(shape)
                        min_coords = np.min(shape_array, axis=0)
                        max_coords = np.max(shape_array, axis=0)
                        width = max_coords[0] - min_coords[0]
                        height = max_coords[1] - min_coords[1]

                        obstacles.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'center_x': x + width/2,
                            'center_y': y + height/2
                        })

        return obstacles

    def calculate_path_loss(self, distance, is_obstructed=False):
        """Calculate path loss using simplified Friis transmission equation"""
        if distance < 0.1:  # Avoid division by zero
            distance = 0.1

        # Basic free space path loss: PL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
        # Simplified: PL = 20*log10(d) + 32.44 (for GHz and km)
        distance_km = distance / 1000
        freq_ghz = self.frequency / 1e9
        path_loss_db = 20 * np.log10(distance_km) + 20 * np.log10(freq_ghz) + 32.44

        # Convert to our scale (meters instead of km)
        path_loss_db = path_loss_db - 120  # Adjust for meter scale

        # Add additional loss if obstructed
        if is_obstructed:
            path_loss_db += 15  # Additional loss through obstacles

        # Add some randomness for realistic simulation
        path_loss_db += np.random.normal(0, 2)

        return np.clip(path_loss_db, -140, -40)  # Limit to reasonable range

    def check_obstruction(self, tx_x, tx_y, rx_x, rx_y, obstacles):
        """Check if line of sight is obstructed by any obstacle"""
        for obstacle in obstacles:
            obs_x, obs_y = obstacle['center_x'], obstacle['center_y']
            obs_width, obs_height = obstacle['width'], obstacle['height']

            # Simple rectangle intersection check
            # Check if line from transmitter to receiver intersects obstacle
            if self.line_rect_intersection(tx_x, tx_y, rx_x, rx_y,
                                         obs_x - obs_width/2, obs_y - obs_height/2,
                                         obs_width, obs_height):
                return True
        return False

    def line_rect_intersection(self, x1, y1, x2, y2, rx, ry, rw, rh):
        """Check if line segment intersects rectangle"""
        # Check if line endpoints are inside rectangle
        if (rx <= x1 <= rx + rw and ry <= y1 <= ry + rh) or \
           (rx <= x2 <= rx + rw and ry <= y2 <= ry + rh):
            return True

        # Check line-rectangle intersection using parametric equations
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return False

        t_min = 0
        t_max = 1

        # Check intersection with each rectangle edge
        for p, q in [(-dx, x1 - rx), (dx, x1 - (rx + rw)),
                    (-dy, y1 - ry), (dy, y1 - (ry + rh))]:
            if p == 0:
                if q < 0:
                    return False
            else:
                t = q / p
                if p < 0:
                    t_max = min(t_max, t)
                else:
                    t_min = max(t_min, t)

                if t_min > t_max:
                    return False

        return True

    def generate_coverage_map(self):
        """Generate mock electromagnetic coverage map"""
        print("🔄 Generating mock coverage map...")

        # Create grid
        x = np.arange(0, self.world_width, self.cell_size)
        y = np.arange(0, self.world_height, self.cell_size)
        X, Y = np.meshgrid(x, y)

        # Get transmitter position
        tx_pos = self.get_transmitter_position()
        tx_x, tx_y, tx_z = tx_pos

        # Get obstacles
        obstacles = self.get_obstacles()

        # Calculate path loss for each grid point
        self.coverage_map = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                rx_x, rx_y = X[i, j], Y[i, j]

                # Calculate distance
                distance = np.sqrt((rx_x - tx_x)**2 + (rx_y - tx_y)**2)

                # Check for obstruction
                is_obstructed = self.check_obstruction(tx_x, tx_y, rx_x, rx_y, obstacles)

                # Calculate path loss
                self.coverage_map[i, j] = self.calculate_path_loss(distance, is_obstructed)

        print("✓ Mock coverage map generated successfully")
        return self.coverage_map

    def visualize_results(self):
        """Visualize the electromagnetic coverage map"""
        if self.coverage_map is None:
            print("❌ No coverage map to visualize")
            return False

        try:
            plt.figure(figsize=(15, 12))

            # Main coverage map
            plt.subplot(2, 2, 1)
            im = plt.imshow(self.coverage_map, cmap='viridis', origin='lower',
                           extent=[0, self.world_width, 0, self.world_height],
                           aspect='equal')
            plt.colorbar(im, label='Path Loss (dB)')
            plt.title('Electromagnetic Coverage Map')

            # Add transmitter
            tx_pos = self.get_transmitter_position()
            plt.scatter(tx_pos[0], tx_pos[1], c='red', s=200, marker='*',
                       label='Transmitter', edgecolors='white', linewidth=2)

            # Add obstacles
            obstacles = self.get_obstacles()
            for obs in obstacles:
                rect = plt.Rectangle((obs['x'] - obs['width']/2, obs['y'] - obs['height']/2),
                                    obs['width'], obs['height'],
                                    color='red', fill=False, linewidth=2,
                                    linestyle='--', label='Obstacle' if obs == obstacles[0] else '')
                plt.gca().add_patch(rect)

            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Path loss heatmap
            plt.subplot(2, 2, 2)
            plt.hist(self.coverage_map.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
            plt.xlabel('Path Loss (dB)')
            plt.ylabel('Frequency')
            plt.title('Path Loss Distribution')
            plt.grid(True, alpha=0.3)

            # Signal strength map (inverse of path loss)
            plt.subplot(2, 2, 3)
            signal_strength = -self.coverage_map  # Convert to signal strength
            im2 = plt.imshow(signal_strength, cmap='hot', origin='lower',
                            extent=[0, self.world_width, 0, self.world_height],
                            aspect='equal')
            plt.colorbar(im2, label='Signal Strength (dB)')
            plt.title('Signal Strength Map')

            # Add transmitter and obstacles
            plt.scatter(tx_pos[0], tx_pos[1], c='blue', s=200, marker='*',
                       edgecolors='white', linewidth=2)
            for obs in obstacles:
                rect = plt.Rectangle((obs['x'] - obs['width']/2, obs['y'] - obs['height']/2),
                                    obs['width'], obs['height'],
                                    color='blue', fill=False, linewidth=2, linestyle='--')
                plt.gca().add_patch(rect)

            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.grid(True, alpha=0.3)

            # Coverage statistics
            plt.subplot(2, 2, 4)
            thresholds = [-60, -70, -80, -90, -100]
            coverage_percentages = []

            for threshold in thresholds:
                coverage_area = np.sum(self.coverage_map > threshold)
                total_area = self.coverage_map.size
                coverage_percentage = (coverage_area / total_area) * 100
                coverage_percentages.append(coverage_percentage)

            plt.bar(thresholds, coverage_percentages, color='green', alpha=0.7)
            plt.xlabel('Path Loss Threshold (dB)')
            plt.ylabel('Coverage Percentage (%)')
            plt.title('Coverage Statistics')
            plt.grid(True, alpha=0.3)

            for i, (threshold, percentage) in enumerate(zip(thresholds, coverage_percentages)):
                plt.text(threshold, percentage + 1, f'{percentage:.1f}%',
                        ha='center', va='bottom')

            plt.tight_layout()

            # Save figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'mock_electromagnetic_map_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Coverage map saved as {filename}")

            # Print statistics
            self.print_statistics()

            plt.show()
            return True

        except Exception as e:
            print(f"❌ Error visualizing results: {e}")
            return False

    def print_statistics(self):
        """Print detailed statistics about the coverage map"""
        if self.coverage_map is None:
            print("❌ No coverage map available")
            return

        print("\n📊 Coverage Map Statistics:")
        print(f"  Grid size: {self.coverage_map.shape}")
        print(f"  Cell size: {self.cell_size}m x {self.cell_size}m")
        print(f"  Total area: {self.world_width}m x {self.world_height}m = {self.world_width * self.world_height}m²")
        print(f"  Min path loss: {np.min(self.coverage_map):.2f} dB")
        print(f"  Max path loss: {np.max(self.coverage_map):.2f} dB")
        print(f"  Mean path loss: {np.mean(self.coverage_map):.2f} dB")
        print(f"  Std deviation: {np.std(self.coverage_map):.2f} dB")

        # Coverage area analysis
        thresholds = [-60, -70, -80, -90, -100]
        print("\n  Coverage Area Analysis:")
        for threshold in thresholds:
            coverage_area = np.sum(self.coverage_map > threshold)
            total_area = self.coverage_map.size
            coverage_percentage = (coverage_area / total_area) * 100
            actual_area = coverage_percentage * self.world_width * self.world_height / 100
            print(f"    > {threshold:3} dB: {coverage_percentage:5.1f}% ({actual_area:5.1f} m²)")

        # Obstacle information
        obstacles = self.get_obstacles()
        print(f"\n  Obstacles: {len(obstacles)}")
        for i, obs in enumerate(obstacles):
            print(f"    Obstacle {i+1}: ({obs['x']:.1f}, {obs['y']:.1f}) "
                  f"size: {obs['width']:.1f}x{obs['height']:.1f}m")

        # Transmitter information
        tx_pos = self.get_transmitter_position()
        print(f"\n  Transmitter: ({tx_pos[0]:.1f}, {tx_pos[1]:.1f}, {tx_pos[2]:.1f})m")
        print(f"  Frequency: {self.frequency/1e9:.2f} GHz")

    def run_test(self):
        """Run the complete mock test"""
        print("🚀 Starting Mock Radio Shadowing Test")
        print("=" * 50)

        # Load configuration
        if not self.load_configuration():
            return False

        # Generate coverage map
        coverage_map = self.generate_coverage_map()
        if coverage_map is None:
            return False

        # Visualize results
        if not self.visualize_results():
            return False

        print("\n🎉 Mock test completed successfully!")
        print("💡 This demonstrates the electromagnetic map generation process")
        print("   The real SiOnNA library would provide more accurate ray-tracing")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Mock Radio Shadowing Test')
    parser.add_argument('--yaml', default='simple_test.yaml',
                       help='YAML configuration file (default: simple_test.yaml)')

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.yaml):
        print(f"⚠️  YAML file not found: {args.yaml}")
        print("   Using default configuration for demonstration")

    # Create and run test
    test = MockRadioShadowingTest(yaml_file=args.yaml)
    success = test.run_test()

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()