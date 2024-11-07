import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

# Configure figure with a corporate style
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')  # Background color
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine
ax.spines['left'].set_linewidth(1.5)  # Thicker left spine
ax.spines['bottom'].set_linewidth(1.5)  # Thicker bottom spine

# Set axis limits
ax.set_xlim(-100, 100)
ax.set_ylim(-20, 100)
ax.set_xticks(np.arange(-100, 101, 20))  # X ticks
ax.set_yticks(np.arange(-10, 110, 10))  # Y ticks

# Set the aspect of the axes to be equal
ax.set_aspect('equal')  # This makes sure the circle is drawn as a circle

# Add a dashed light gray line at y=0
ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)

# Load car icon image
ego_car_img = mpimg.imread('car_icon.png')  # Ensure you have a car icon image named 'car_icon.png'
target_car_img = mpimg.imread('target_car.jpg')  # Ensure you have a target car image named 'target_car.jpg'

# Draw the target car image (moved closer) with increased size
target_car_x = -20  # Centered horizontally
target_car_y = 43  # Adjusted position for the target car, to fit the new Y limits
target_car_image = ax.imshow(target_car_img, extent=[target_car_x, target_car_x + 28, target_car_y, target_car_y + 15], aspect='auto')

# Create a damping level text bar
damping_text = ax.text(0, 100, 'Damping Level: 0 dB', fontsize=14, fontweight='bold', ha='center')

# Initialize line and text for max visibility
max_visibility_line = None
visibility_text = None

# Update function for the animation
def update(frame):
    global max_visibility_line, visibility_text  # Use global variables for the line and text

    # Clear previous patches
    for patch in ax.patches:
        patch.remove()  # Remove existing patches

    # Define ego car position (set to 0 on Y-axis)
    ego_car_x = -13  # Centered with respect to the red circle, moved back
    ego_car_y = -1  # Position of the car (on the 0 line)

    # Draw the ego car image (increased size)
    ax.imshow(ego_car_img, extent=[ego_car_x, ego_car_x + 28, ego_car_y, ego_car_y + 15], aspect='auto')

    # Create new radar detection area starting from the red circle
    radar_circle = plt.Circle((1, 14), 1.25, color='red', alpha=1)  # Positioned just above the car
    ax.add_patch(radar_circle)

    # Define radius for radar detection area based on damping level
    radius = max(40, 80 - (frame * 0.5))  # Adjusted to reach a minimum radius more quickly

    # Update damping level text
    damping_level = min(30, (frame // 10) * 5)  # Max at 30 dB
    damping_text.set_text(f'Damping Level: {damping_level} dB')

    # Create new radar detection area starting from the red circle
    radar_detection_area = plt.Polygon(
        [(1, 14), (radius, radius - 2), (-radius, radius - 2)],
        color='blue', alpha=0.3
    )
    ax.add_patch(radar_detection_area)

    # Add or update the max visibility line
    max_visibility_line_y = radius - 2  # Y position of the max visibility line
    if max_visibility_line is None:
        # Create line if it does not exist
        max_visibility_line, = ax.plot([-radius, radius], [max_visibility_line_y, max_visibility_line_y], color='green',
                                       linewidth=2)
    else:
        # Update line position and length
        max_visibility_line.set_xdata([-radius, radius])
        max_visibility_line.set_ydata([max_visibility_line_y, max_visibility_line_y])

        # Define the number of classes and maximum visibility range
        num_classes = 7

        # Calculate the class index based on the current frame
        class_index = min(num_classes - 1, int((frame / 50) * num_classes))  # Assuming 90 frames

        # Calculate target opacity based on the class index
        target_opacity = max(0, (num_classes - class_index) / num_classes)  # Opacity goes from 1 to 0

        # Set the opacity of the target car image
        target_car_image.set_alpha(target_opacity)

    # Add or update text for max visibility
    if visibility_text is None:
        visibility_text = ax.text(radius, max_visibility_line_y + 0.2, 'Max Visibility', fontsize=12, ha='left',
                                  color='green')
    else:
        visibility_text.set_position((radius, max_visibility_line_y + 0.2))

    return radar_detection_area, damping_text, radar_circle, max_visibility_line, visibility_text

# Set axis labels
ax.set_xlabel('Lateral distance (m)', fontsize=12)
ax.set_ylabel('Longitudinal distance (m)', fontsize=12)

# Create animation
anim = animation.FuncAnimation(fig, update, frames=90, interval=1000, blit=False, repeat=True)

# Save the animation as a GIF
anim_path = 'radar_detection_area_animation_aligned.gif'
anim.save(anim_path, writer='pillow', fps=10)

print(f"GIF saved as {anim_path}")
