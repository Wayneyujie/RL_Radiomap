#!/usr/bin/env python3
"""
Interactive Map Editor
Supports drag-and-drop drawing of rectangular obstacles, setting robot positions and goals,
and saving to YAML format compatible with EnvBase
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import yaml
from collections import defaultdict




class MapEditor:
    def __init__(self):
        """Initialize map editor"""
        # Map data (default values)
        self.world_width = 20.0
        self.world_height = 20.0
        self.obstacles = []  # Store obstacles: [(center_x, center_y, width, height), ...]
        
        # 多机器人支持：改为机器人列表
        self.robots = []  # Store robots: [{'start': [x, y], 'goal': [x, y], 'cooperative': bool}, ...]
        self.current_robot_idx = 0  # Currently selected robot index
        
        # Try to load previous YAML file
        self.load_yaml()
        
        # 如果没有加载到机器人，添加一个默认机器人
        if len(self.robots) == 0:
            self.robots.append({
                'start': [2.0, 2.0],
                'goal': [18.0, 18.0],
                'cooperative': True
            })
            self.current_robot_idx = 0
        
        # Interaction state
        self.drawing_mode = False  # Whether currently drawing obstacles
        self.setting_start = False  # Whether setting start position
        self.setting_goal = False  # Whether setting goal position
        self.drag_start_pos = None  # Drag start position
        self.current_rect = None  # Currently drawing rectangle
        
        # Create figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Draw initial map
        self.update_display()
        
        # Create toolbar buttons (after loading data so text boxes show correct values)
        self.create_buttons()
        
        plt.tight_layout()
        plt.show()
    
    def create_buttons(self):
        """Create toolbar buttons"""
        # Button position and size
        button_width = 0.10
        button_height = 0.05
        button_y = 0.02
        button_spacing = 0.11
        
        # Add obstacle button
        ax_add = plt.axes([0.02, button_y, button_width, button_height])
        self.btn_add = Button(ax_add, 'Add Obstacle\n(Drag)')
        self.btn_add.on_clicked(self.enable_draw_mode)
        
        # Add robot button
        ax_add_robot = plt.axes([0.02 + button_spacing, button_y, button_width, button_height])
        self.btn_add_robot = Button(ax_add_robot, 'Add Robot')
        self.btn_add_robot.on_clicked(self.add_robot)
        
        # Delete robot button
        ax_del_robot = plt.axes([0.02 + 2*button_spacing, button_y, button_width, button_height])
        self.btn_del_robot = Button(ax_del_robot, 'Delete Robot')
        self.btn_del_robot.on_clicked(self.delete_current_robot)
        
        # Set start position button
        ax_start = plt.axes([0.02 + 3*button_spacing, button_y, button_width, button_height])
        self.btn_start = Button(ax_start, 'Set Start')
        self.btn_start.on_clicked(self.enable_set_start)
        
        # Set goal position button
        ax_goal = plt.axes([0.02 + 4*button_spacing, button_y, button_width, button_height])
        self.btn_goal = Button(ax_goal, 'Set Goal')
        self.btn_goal.on_clicked(self.enable_set_goal)
        
        # Clear all button
        ax_clear = plt.axes([0.02 + 5*button_spacing, button_y, button_width, button_height])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_all)
        
        # Save YAML button
        ax_save = plt.axes([0.02 + 6*button_spacing, button_y, button_width, button_height])
        self.btn_save = Button(ax_save, 'Save YAML')
        self.btn_save.on_clicked(self.save_yaml)
        
        # Robot selector
        ax_robot_label = plt.axes([0.02 + 7*button_spacing, button_y + 0.03, 0.06, 0.02])
        ax_robot_label.axis('off')
        ax_robot_label.text(0.5, 0.5, 'Robot:', ha='center', va='center', fontsize=8)
        
        ax_robot = plt.axes([0.02 + 7*button_spacing, button_y, 0.06, button_height])
        self.text_robot = TextBox(ax_robot, '', initial=str(self.current_robot_idx + 1))
        self.text_robot.on_submit(self.select_robot)
        
        # World size input boxes
        ax_width_label = plt.axes([0.02 + 8*button_spacing, button_y + 0.03, 0.06, 0.02])
        ax_width_label.axis('off')
        ax_width_label.text(0.5, 0.5, 'Width:', ha='center', va='center', fontsize=8)
        
        ax_width = plt.axes([0.02 + 8*button_spacing, button_y, 0.06, button_height])
        self.text_width = TextBox(ax_width, '', initial=str(self.world_width))
        self.text_width.on_submit(self.update_width)
        
        ax_height_label = plt.axes([0.02 + 9*button_spacing, button_y + 0.03, 0.06, 0.02])
        ax_height_label.axis('off')
        ax_height_label.text(0.5, 0.5, 'Height:', ha='center', va='center', fontsize=8)
        
        ax_height = plt.axes([0.02 + 9*button_spacing, button_y, 0.06, button_height])
        self.text_height = TextBox(ax_height, '', initial=str(self.world_height))
        self.text_height.on_submit(self.update_height)
    
    def update_width(self, text):
        """Update world width"""
        try:
            self.world_width = float(text)
            self.update_display()
        except ValueError:
            pass
    
    def update_height(self, text):
        """Update world height"""
        try:
            self.world_height = float(text)
            self.update_display()
        except ValueError:
            pass
    
    def enable_draw_mode(self, event):
        """Enable drawing mode"""
        self.drawing_mode = True
        self.setting_start = False
        self.setting_goal = False
        print("Drag mode enabled: Drag on canvas to draw rectangular obstacles")
    
    def enable_set_start(self, event):
        """Enable set start position mode"""
        self.setting_start = True
        self.setting_goal = False
        self.drawing_mode = False
        print(f"Click on canvas to set robot {self.current_robot_idx + 1} start position")
    
    def enable_set_goal(self, event):
        """Enable set goal position mode"""
        self.setting_goal = True
        self.setting_start = False
        self.drawing_mode = False
        print(f"Click on canvas to set robot {self.current_robot_idx + 1} goal position")
    
    def add_robot(self, event):
        """Add a new robot"""
        # 添加新机器人，使用默认位置（避免重叠）
        offset = len(self.robots) * 2.0
        new_start = [2.0 + offset, 2.0 + offset]
        new_goal = [18.0 - offset, 18.0 - offset]
        self.robots.append({
            'start': new_start,
            'goal': new_goal,
            'cooperative': True
        })
        self.current_robot_idx = len(self.robots) - 1
        self.text_robot.set_val(str(self.current_robot_idx + 1))
        self.update_display()
        print(f"Added robot {len(self.robots)}. Total robots: {len(self.robots)}")
    
    def delete_current_robot(self, event):
        """Delete the currently selected robot"""
        if len(self.robots) <= 1:
            print("Cannot delete: at least one robot is required")
            return
        
        self.robots.pop(self.current_robot_idx)
        if self.current_robot_idx >= len(self.robots):
            self.current_robot_idx = len(self.robots) - 1
        self.text_robot.set_val(str(self.current_robot_idx + 1))
        self.update_display()
        print(f"Deleted robot. Total robots: {len(self.robots)}")
    
    def select_robot(self, text):
        """Select robot by index"""
        try:
            idx = int(text) - 1
            if 0 <= idx < len(self.robots):
                self.current_robot_idx = idx
                self.update_display()
                print(f"Selected robot {idx + 1}")
            else:
                self.text_robot.set_val(str(self.current_robot_idx + 1))
                print(f"Invalid robot index. Use 1-{len(self.robots)}")
        except ValueError:
            self.text_robot.set_val(str(self.current_robot_idx + 1))
    
    def clear_all(self, event):
        """Clear all obstacles"""
        self.obstacles = []
        self.update_display()
        print("All obstacles cleared")
    
    def on_mouse_press(self, event):
        """Handle mouse press event"""
        if event.inaxes != self.ax:
            return
        
        # Check if right click (delete obstacle or robot)
        if event.button == 3:  # Right button
            # 先检查是否点击了机器人
            if self.delete_robot_at(event.xdata, event.ydata):
                return
            # 否则删除障碍物
            self.delete_obstacle_at(event.xdata, event.ydata)
            return
        
        # Left click
        if event.button == 1:
            if self.drawing_mode:
                # Start drawing obstacle
                self.drag_start_pos = (event.xdata, event.ydata)
            elif self.setting_start:
                # Set current robot's start position
                current_robot = self.robots[self.current_robot_idx]
                current_robot['start'] = [event.xdata, event.ydata]
                self.setting_start = False
                self.update_display()
                print(f"Robot {self.current_robot_idx + 1} start set to: ({event.xdata:.2f}, {event.ydata:.2f})")
            elif self.setting_goal:
                # Set current robot's goal position
                current_robot = self.robots[self.current_robot_idx]
                current_robot['goal'] = [event.xdata, event.ydata]
                self.setting_goal = False
                self.update_display()
                print(f"Robot {self.current_robot_idx + 1} goal set to: ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def on_mouse_move(self, event):
        """Handle mouse move event"""
        if event.inaxes != self.ax:
            return
        
        if self.drawing_mode and self.drag_start_pos is not None:
            # Update rectangle being drawn in real-time
            start_x, start_y = self.drag_start_pos
            current_x, current_y = event.xdata, event.ydata
            
            width = abs(current_x - start_x)
            height = abs(current_y - start_y)
            x = min(start_x, current_x)
            y = min(start_y, current_y)
            
            # Remove previous temporary rectangle
            if self.current_rect is not None:
                self.current_rect.remove()
            
            # Create new temporary rectangle
            self.current_rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
            )
            self.ax.add_patch(self.current_rect)
            self.fig.canvas.draw_idle()
    
    def on_mouse_release(self, event):
        """Handle mouse release event"""
        if event.inaxes != self.ax:
            return
        
        if self.drawing_mode and self.drag_start_pos is not None:
            # Finish drawing obstacle
            start_x, start_y = self.drag_start_pos
            end_x, end_y = event.xdata, event.ydata
            
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)
            
            # Minimum size check
            if width > 0.1 and height > 0.1:
                center_x = (start_x + end_x) / 2
                center_y = (start_y + end_y) / 2
                
                # Add obstacle
                self.obstacles.append((center_x, center_y, width, height))
                print(f"Obstacle added: center({center_x:.2f}, {center_y:.2f}), size({width:.2f}, {height:.2f})")
            
            # Clean up temporary rectangle
            if self.current_rect is not None:
                self.current_rect.remove()
                self.current_rect = None
            
            self.drag_start_pos = None
            self.update_display()
    
    def delete_robot_at(self, x, y):
        """Delete robot at specified position (right-click)"""
        for i, robot in enumerate(self.robots):
            start = robot['start']
            goal = robot['goal']
            # 检查是否点击了起点或终点附近
            if (abs(start[0] - x) < 0.5 and abs(start[1] - y) < 0.5) or \
               (abs(goal[0] - x) < 0.5 and abs(goal[1] - y) < 0.5):
                if len(self.robots) <= 1:
                    print("Cannot delete: at least one robot is required")
                    return True
                self.robots.pop(i)
                if self.current_robot_idx >= len(self.robots):
                    self.current_robot_idx = len(self.robots) - 1
                self.text_robot.set_val(str(self.current_robot_idx + 1))
                self.update_display()
                print(f"Robot {i+1} deleted. Total robots: {len(self.robots)}")
                return True
        return False
    
    def delete_obstacle_at(self, x, y):
        """Delete obstacle at specified position"""
        for i, (cx, cy, w, h) in enumerate(self.obstacles):
            if (cx - w/2 <= x <= cx + w/2) and (cy - h/2 <= y <= cy + h/2):
                self.obstacles.pop(i)
                self.update_display()
                print(f"Obstacle {i+1} deleted")
                return
    
    def update_display(self):
        """Update display"""
        self.ax.clear()
        
        # Set axis limits
        self.ax.set_xlim(0, self.world_width)
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('Map Editor - Multi-Robot Support', fontsize=14, fontweight='bold')
        
        # Draw obstacles
        for i, (cx, cy, w, h) in enumerate(self.obstacles):
            rect = patches.Rectangle(
                (cx - w/2, cy - h/2), w, h,
                linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5
            )
            self.ax.add_patch(rect)
            # Show obstacle number
            self.ax.text(cx, cy, str(i+1), ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        # Draw all robots with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.robots), 10)))
        for i, robot in enumerate(self.robots):
            start = robot['start']
            goal = robot['goal']
            color = colors[i]
            
            # 当前选中的机器人用更粗的边框
            linewidth = 3 if i == self.current_robot_idx else 2
            alpha = 0.9 if i == self.current_robot_idx else 0.6
            
            # Draw start position
            self.ax.scatter(start[0], start[1], 
                           s=300, c=color, marker='o', 
                           edgecolors='darkgreen', linewidth=linewidth, 
                           zorder=10, alpha=alpha, 
                           label=f'Robot {i+1} Start' if i == 0 else '')
            
            # Draw goal position
            self.ax.scatter(goal[0], goal[1], 
                           s=300, c=color, marker='*', 
                           edgecolors='darkred', linewidth=linewidth, 
                           zorder=10, alpha=alpha,
                           label=f'Robot {i+1} Goal' if i == 0 else '')
            
            # 显示机器人编号
            self.ax.text(start[0], start[1] - 0.8, f'R{i+1}S', 
                        ha='center', va='top', fontsize=8, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            self.ax.text(goal[0], goal[1] + 0.8, f'R{i+1}G', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Display information
        info_text = f'Obstacles: {len(self.obstacles)}\nRobots: {len(self.robots)}\nWorld: {self.world_width:.1f}m × {self.world_height:.1f}m'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 显示当前选中的机器人信息
        if len(self.robots) > 0:
            current = self.robots[self.current_robot_idx]
            robot_info = f'Current: Robot {self.current_robot_idx + 1}\nStart: ({current["start"][0]:.1f}, {current["start"][1]:.1f})\nGoal: ({current["goal"][0]:.1f}, {current["goal"][1]:.1f})'
            self.ax.text(0.98, 0.98, robot_info, transform=self.ax.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()
    
    def _format_number(self, value):
        """Format number: return int if whole number, float otherwise"""
        if isinstance(value, (int, float)):
            # Check if it's a whole number
            if isinstance(value, float) and value.is_integer():
                return int(value)
            elif isinstance(value, int):
                return value
            else:
                return float(value)
        else:
            return value
    
    def load_yaml(self, filename='map_editor_output.yaml'):
        """Load YAML file and populate editor with data"""
        import os
        
        if not os.path.exists(filename):
            print(f"Info: {filename} not found, using default values")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                print(f"Warning: {filename} is empty, using default values")
                return False
            
            # Load world dimensions
            if 'world' in yaml_data:
                world = yaml_data['world']
                if 'width' in world:
                    self.world_width = float(world['width'])
                if 'height' in world:
                    self.world_height = float(world['height'])
            
            # Load robot positions
            if 'robots' in yaml_data:
                robots = yaml_data['robots']
                if 'state' in robots and len(robots['state']) >= 2:
                    self.robot_start = [float(robots['state'][0]), float(robots['state'][1])]
                if 'goal' in robots and len(robots['goal']) >= 2:
                    self.robot_goal = [float(robots['goal'][0]), float(robots['goal'][1])]
            
            # Load obstacles
            self.obstacles = []
            if 'obstacles' in yaml_data and len(yaml_data['obstacles']) > 0:
                obstacle_group = yaml_data['obstacles'][0]
                if 'distribute' in obstacle_group:
                    distribute = obstacle_group['distribute']
                    states = distribute.get('states', [])
                    shapes = distribute.get('shapes', [])
                    
                    # Convert YAML format to editor format
                    for state, shape in zip(states, shapes):
                        if len(state) >= 2 and len(shape) >= 3:
                            # Extract position from state (this is where shape's [0,0] vertex is placed)
                            state_x = float(state[0])
                            state_y = float(state[1])
                            
                            # Extract width and height from shape
                            # Shape format: [[0,0], [w,0], [w,h], [0,h]]
                            # Calculate bounding box
                            x_coords = [vertex[0] for vertex in shape]
                            y_coords = [vertex[1] for vertex in shape]
                            
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            
                            # Calculate actual center position
                            # Since shape starts from [0,0], its center is at (w/2, h/2)
                            # So obstacle center = state position + shape center offset
                            shape_center_x = width / 2
                            shape_center_y = height / 2
                            center_x = state_x + shape_center_x
                            center_y = state_y + shape_center_y
                            
                            # Add obstacle with center position
                            self.obstacles.append((center_x, center_y, width, height))
            
            print(f"✓ Successfully loaded {filename}")
            print(f"   - World size: {self.world_width}m × {self.world_height}m")
            print(f"   - Obstacles: {len(self.obstacles)}")
            print(f"   - Start position: ({self.robot_start[0]:.2f}, {self.robot_start[1]:.2f})")
            print(f"   - Goal position: ({self.robot_goal[0]:.2f}, {self.robot_goal[1]:.2f})")
            return True
            
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse {filename}: {e}")
            print("Using default values")
            return False
        except Exception as e:
            print(f"Error: Failed to load {filename}: {e}")
            print("Using default values")
            return False
    
    def _to_python_type(self, value):
        """Convert numpy types to Python native types"""
        if isinstance(value, (np.integer, np.floating)):
            # Convert numpy numbers to Python int/float
            if isinstance(value, np.integer):
                return int(value)
            else:
                return float(value)
        elif isinstance(value, np.ndarray):
            # Convert numpy array to Python list
            return [self._to_python_type(x) for x in value.tolist()]
        elif isinstance(value, (list, tuple)):
            # Recursively convert list/tuple elements
            return [self._to_python_type(x) for x in value]
        else:
            return value
    
    def save_yaml(self, event):
        """Save as YAML file"""
        # Convert all values to Python native types
        world_height = int(self.world_height) if self.world_height == int(self.world_height) else float(self.world_height)
        world_width = int(self.world_width) if self.world_width == int(self.world_width) else float(self.world_width)
        
        # Build YAML data structure with Python native types
        yaml_data = {
            'world': {
                'height': int(world_height),
                'width': int(world_width),
                'step_time': 0.1,
                'sample_time': 0.1,
                'offset': [0, 0]
            },
            'robots': {
                'type': 'robot_acker',
                'number': 1,
                'state': [self._format_number(self.robot_start[0]), 
                         self._format_number(self.robot_start[1]), 
                         0.5, 0],
                'shape': [0.322, 0.22, 0.2, 0.22],
                'goal': [self._format_number(self.robot_goal[0]), 
                        self._format_number(self.robot_goal[1]), 
                        0],
                'vel_type': 'steer',
                'vel_min': [-1, -1],
                'vel_max': [1, 1],
                'psi_limit': 1,
                'arrive_mode': 'state',
                'edgecolor': 'b'
            }
        }
        
        # Process obstacles
        if len(self.obstacles) > 0:
            states = []
            shapes = []
            
            for cx, cy, w, h in self.obstacles:
                # Convert to Python native types
                cx = float(cx)
                cy = float(cy)
                w = float(w)
                h = float(h)
                
                # State: obstacle center position [x, y, 0]
                # Note: If shapes start from [0,0], states should be bottom-left corner
                # But if states is center, shapes should be relative to center
                # Based on simple_test.yaml analysis: states appears to be the position where
                # shape's [0,0] vertex is placed. So we need to adjust states to account for shape center offset.
                
                # Calculate shape center offset (since shape starts from [0,0], center is at [w/2, h/2])
                shape_center_x = w / 2
                shape_center_y = h / 2
                
                # Adjust states to be the bottom-left corner position
                # So that when shape's [0,0] is placed at states, the center ends up at (cx, cy)
                state_x = cx - shape_center_x
                state_y = cy - shape_center_y
                states.append([state_x, state_y, 0])
                
                # Shape: rectangular polygon vertices starting from (0,0)
                # Format consistent with simple_test.yaml: starts from (0,0), extends right and up
                shape = [
                    [0, 0],      # Bottom-left (start point)
                    [w, 0],      # Bottom-right
                    [w, h],      # Top-right
                    [0, h]       # Top-left
                ]
                shapes.append(shape)
            
            # Use compact format for obstacles like simple_test.yaml
            yaml_data['obstacles'] = [{
                'type': 'obstacle_polygon',
                'number': len(self.obstacles),
                'distribute': {
                    'mode': 'manual',
                    'states': states,
                    'shapes': shapes
                }
            }]
        else:
            yaml_data['obstacles'] = []
        
        # Save file with custom dumper
        filename = 'map_editor_output.yaml'
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write world section
                f.write('world:\n')
                f.write(f'  height: {yaml_data["world"]["height"]}\n')
                f.write(f'  width: {yaml_data["world"]["width"]}\n')
                f.write(f'  step_time: {yaml_data["world"]["step_time"]}\n')
                f.write(f'  sample_time: {yaml_data["world"]["sample_time"]}\n')
                f.write(f'  offset: {yaml_data["world"]["offset"]}\n')
                f.write('\n')
                
                # Write robots section
                f.write('robots:\n')
                f.write(f"  type: '{yaml_data['robots']['type']}'  # robot_omni, robot_diff, robot_acker\n")
                f.write(f"  number: {yaml_data['robots']['number']}\n")
                # Format state array with proper number types (int if whole number)
                state_list = yaml_data['robots']['state']
                state_str = '[' + ', '.join(str(self._format_number(x)) for x in state_list) + ']'
                f.write(f"  state: {state_str}  # Starting position [x, y, theta, velocity]\n")
                f.write(f"  shape: {yaml_data['robots']['shape']} # for acker (default)\n")
                # Format goal array with proper number types (int if whole number)
                goal_list = yaml_data['robots']['goal']
                goal_str = '[' + ', '.join(str(self._format_number(x)) for x in goal_list) + ']'
                f.write(f"  goal: {goal_str}  # Target position\n")
                f.write(f"  vel_type: '{yaml_data['robots']['vel_type']}'\n")
                f.write(f"  vel_min: {yaml_data['robots']['vel_min']}\n")
                f.write(f"  vel_max: {yaml_data['robots']['vel_max']}\n")
                f.write(f"  psi_limit: {yaml_data['robots']['psi_limit']}\n")
                f.write(f"  arrive_mode: '{yaml_data['robots']['arrive_mode']}'\n")
                f.write(f"  edgecolor: '{yaml_data['robots']['edgecolor']}'\n")
                f.write('\n')
                
                # Write obstacles section
                f.write('obstacles:\n')
                if len(self.obstacles) > 0:
                    f.write('  # Obstacles from map editor\n')
                    f.write("  - type: 'obstacle_polygon'\n")
                    f.write(f"    number: {yaml_data['obstacles'][0]['number']}\n")
                    f.write("    distribute: {mode: 'manual',\n")
                    f.write("              states:\n")
                    f.write(f"              {yaml_data['obstacles'][0]['distribute']['states']},\n")
                    f.write("              shapes: [\n")
                    for i, shape in enumerate(yaml_data['obstacles'][0]['distribute']['shapes']):
                        # Format shape with comments like simple_test.yaml
                        if i == 0:
                            f.write("              # Obstacle shape\n")
                        f.write(f"              {shape}")
                        if i < len(yaml_data['obstacles'][0]['distribute']['shapes']) - 1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("              ] }\n")
                else:
                    f.write("  []\n")
            
            print(f"\n✓ YAML file saved: {filename}")
            print(f"   - World size: {self.world_width}m × {self.world_height}m")
            print(f"   - Number of obstacles: {len(self.obstacles)}")
            print(f"   - Start position: ({self.robot_start[0]:.2f}, {self.robot_start[1]:.2f})")
            print(f"   - Goal position: ({self.robot_goal[0]:.2f}, {self.robot_goal[1]:.2f})")
        except Exception as e:
            print(f"\n✗ Save failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    print("=" * 60)
    print("Map Editor")
    print("=" * 60)
    print("Usage:")
    print("1. Click 'Add Obstacle' button, then drag on canvas to draw rectangles")
    print("2. Right-click on obstacles to delete them")
    print("3. Click 'Set Start' or 'Set Goal' button, then click on canvas to set position")
    print("4. Adjust world size (width and height)")
    print("5. Click 'Save YAML' button to save map configuration")
    print("=" * 60)
    
    editor = MapEditor()


if __name__ == '__main__':
    main()

