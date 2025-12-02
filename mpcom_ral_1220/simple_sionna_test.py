#!/usr/bin/env python3
"""
Simple SiOnNA Radio Map Generation Test

This script uses SiOnNA library to generate electromagnetic maps
using our simple_test.xml scene with standard antenna configuration.

Based on the standard SiOnNA workflow from 10.29radio_shadowing.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, PlanarArray, Camera
import os
import sys

# Set GPU configuration
gpu_num = 0  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_simple_radio_map():
    """
    Generate electromagnetic map using SiOnNA with simple_test.xml
    """
    print("🚀 Starting SiOnNA Radio Map Generation")
    print("=" * 50)

    try:
        # Load the simple scene
        print("📂 Loading scene from simple_test.xml...")
        scene = load_scene('simple_test.xml')
        print("✓ Scene loaded successfully")

        # Configure antenna array for all transmitters (standard configuration)
        print("📡 Configuring transmitter antenna array...")
        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V"
        )
        print("✓ TX array configured")

        # Configure antenna array for all receivers (standard configuration)
        print("📡 Configuring receiver antenna array...")
        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross"
        )
        print("✓ RX array configured")

        # Set fixed transmitter position in the simple scene
        # Position: center of the scene with some height
        iot_position = [10, 10, 3]  # [x, y, z] in meters (center of 20x20 scene)
        iot_orientation = [0, 1.57, 1.57]  # Standard orientation

        print(f"📡 Adding transmitter at position {iot_position}")
        print(f"   Orientation: {iot_orientation}")

        # Add transmitter instance to scene
        tx = Transmitter(
            name="simple_tx",
            position=iot_position,
            orientation=iot_orientation,
            color=[1, 0, 0]  # Red color
        )

        scene.add(tx)
        print("✓ Transmitter added to scene")

        # Set scene frequency
        scene.frequency = 2.14e9  # 2.14 GHz in Hz; implicitly updates RadioMaterials
        scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        print(f"📡 Frequency set to {scene.frequency/1e9:.2f} GHz")

        # Generate coverage map
        print("🗺️  Generating coverage map...")
        print("   Scene coverage: 20m x 20m")
        print("   Cell size: 0.5m x 0.5m")
        print("   Expected grid: 40 x 40 cells")

        cm = scene.coverage_map(
            max_depth=7,
            diffraction=True,  # Enable diffraction effects
            cm_cell_size=(0.5, 0.5),  # Grid size of coverage map cells in m
            combining_vec=None,
            precoding_vec=None,
            num_samples=int(5e6)  # Number of ray tracing samples
        )
        print("✓ Coverage map generated successfully")

        # Display coverage map
        print("📊 Displaying coverage map...")
        cm.show()

        # Convert coverage map to numpy array for analysis
        path_gain_raw = cm._path_gain.numpy()
        path_gain_db = 20 * np.log10(np.abs(path_gain_raw))  # Use absolute value to avoid log(0)

        # Handle infinite values by replacing them with a very low value
        path_gain_db = np.nan_to_num(path_gain_db, nan=-150, neginf=-150, posinf=0)

        # Extract 2D array for analysis (remove channel dimension if present)
        if len(path_gain_db.shape) == 3:
            path_gain_2d = path_gain_db[0]  # Take first channel
        else:
            path_gain_2d = path_gain_db

        # Print statistics
        finite_values = path_gain_2d[np.isfinite(path_gain_2d)]
        if len(finite_values) > 0:
            print("\n📊 Coverage Map Statistics:")
            print(f"  Shape: {path_gain_2d.shape}")
            print(f"  Cell size: 0.5m x 0.5m")
            print(f"  Geographic coverage: {path_gain_2d.shape[1]*0.5}m x {path_gain_2d.shape[0]*0.5}m")
            print(f"  Min path gain: {np.min(finite_values):.2f} dB")
            print(f"  Max path gain: {np.max(finite_values):.2f} dB")
            print(f"  Mean path gain: {np.mean(finite_values):.2f} dB")
            print(f"  Std deviation: {np.std(finite_values):.2f} dB")
            print(f"  Valid cells: {len(finite_values)}/{path_gain_2d.size}")
            print(f"  Transmitter position: [10, 10, 3]")
            print(f"  Transmitter grid position: [10/0.5, 10/0.5] = [20, 20]")
        else:
            print("\n⚠️  No valid path gain values found")

        # Save coverage map to .npy file
        output_filename = "simple_radio_map.npy"
        print(f"\n💾 Saving coverage map to {output_filename}...")

        # Use the original path gain format from SiOnna (which is already in the correct format)
        radio_map = path_gain_raw
        np.save(output_filename, radio_map)
        print(f"✓ Radio map saved as {output_filename}")
        print(f"  File size: {os.path.getsize(output_filename)} bytes")

        # Also save as ASCII format for compatibility (save 2D array)
        ascii_filename = "simple_radio_map.txt"
        np.savetxt(ascii_filename, path_gain_2d, fmt='%.6f')
        print(f"✓ Radio map also saved as {ascii_filename}")

        # Generate additional visualization
        create_additional_visualizations(path_gain_2d, iot_position)

        print("\n🎉 SiOnNA radio map generation completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during radio map generation: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def create_additional_visualizations(path_gain_db, transmitter_position):
    """
    Create additional visualizations of the electromagnetic map
    """
    try:
        print("\n📈 Creating additional visualizations...")

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Path gain heatmap
        im1 = axes[0, 0].imshow(path_gain_db, cmap='viridis', origin='lower',
                               extent=[0, 20, 0, 20], aspect='equal')
        axes[0, 0].set_title('Path Gain (dB)')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        plt.colorbar(im1, ax=axes[0, 0])

        # Add transmitter position
        axes[0, 0].scatter(transmitter_position[0], transmitter_position[1],
                          c='red', s=200, marker='*',
                          label=f'Transmitter ({transmitter_position[0]}, {transmitter_position[1]})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Signal strength (inverse of path loss)
        signal_strength = -path_gain_db  # Convert to positive signal strength
        im2 = axes[0, 1].imshow(signal_strength, cmap='hot', origin='lower',
                               extent=[0, 20, 0, 20], aspect='equal')
        axes[0, 1].set_title('Signal Strength (dB)')
        axes[0, 1].set_xlabel('X Position (m)')
        axes[0, 1].set_ylabel('Y Position (m)')
        plt.colorbar(im2, ax=axes[0, 1])

        # Add transmitter position
        axes[0, 1].scatter(transmitter_position[0], transmitter_position[1],
                          c='blue', s=200, marker='*')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Histogram of path gain values
        axes[1, 0].hist(path_gain_db.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Path Gain Distribution')
        axes[1, 0].set_xlabel('Path Gain (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Add statistics to histogram
        mean_val = np.mean(path_gain_db)
        std_val = np.std(path_gain_db)
        axes[1, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f} dB')
        axes[1, 0].axvline(mean_val + std_val, color='orange', linestyle='--', label=f'±1σ: {std_val:.1f} dB')
        axes[1, 0].legend()

        # 4. Coverage analysis
        thresholds = [-80, -90, -100, -110]
        coverage_percentages = []

        for threshold in thresholds:
            coverage_area = np.sum(path_gain_db > threshold)
            total_area = path_gain_db.size
            coverage_percentage = (coverage_area / total_area) * 100
            coverage_percentages.append(coverage_percentage)

        bars = axes[1, 1].bar(range(len(thresholds)), coverage_percentages,
                              color='purple', alpha=0.7)
        axes[1, 1].set_title('Coverage Analysis')
        axes[1, 1].set_xlabel('Path Gain Threshold (dB)')
        axes[1, 1].set_ylabel('Coverage Percentage (%)')
        axes[1, 1].set_xticks(range(len(thresholds)))
        axes[1, 1].set_xticklabels([f'>{t}' for t in thresholds])
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, percentage) in enumerate(zip(bars, coverage_percentages)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{percentage:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        # Save the figure
        output_filename = "simple_radio_analysis.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Additional analysis saved as {output_filename}")

        plt.show()

    except Exception as e:
        print(f"⚠️  Warning: Could not create additional visualizations: {e}")

def main():
    """
    Main function to run the SiOnNA radio map generation
    """
    print("Simple SiOnNA Radio Map Generation Test")
    print("Using scene: simple_test.xml")
    print("Library: SiOnna RT")

    # Check if simple_test.xml exists
    if not os.path.exists('simple_test.xml'):
        print("❌ Error: simple_test.xml not found!")
        print("   Please run 'python yaml_to_xml_converter.py simple_test.yaml' first")
        sys.exit(1)

    # Run the radio map generation
    success = generate_simple_radio_map()

    if success:
        print("\n✅ All operations completed successfully!")
        print("Files generated:")
        print("  - simple_radio_map.npy (SiOnNA radio map)")
        print("  - simple_radio_map.txt (ASCII format)")
        print("  - simple_radio_analysis.png (visualization analysis)")
    else:
        print("\n❌ Radio map generation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()