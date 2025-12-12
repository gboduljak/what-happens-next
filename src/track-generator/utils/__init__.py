from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def compose(fs):
    return reduce(
        lambda f, g: lambda x: f(g(x)),
        fs,
        lambda x: x
    )


def concatenate_gifs(gif_paths: List[str], output_path: Path):
    # Load all frames from each GIF
    gif_frames = []
    durations = []  # Store the duration for each GIF to maintain animation speed

    for gif_path in gif_paths:
        with Image.open(gif_path) as img:
            frames = []
            try:
                while True:
                    frames.append(img.copy())
                    img.seek(img.tell() + 1)
            except EOFError:
                pass  # End of frames
            gif_frames.append(frames)
            durations.append(img.info['duration'])  # Get duration of the GIF

    # Get the width of the first GIF to determine the output width
    width, height = gif_frames[0][0].size

    # Create a list to hold concatenated frames
    concatenated_frames = []

    # Concatenate frames from each GIF
    for frames in gif_frames:
        for frame in frames:
            # Create a new image for each frame of the GIF
            new_frame = Image.new("RGB", (width, height))
            new_frame.paste(frame, (0, 0))
            concatenated_frames.append(new_frame)

    # Save the concatenated frames as a GIF
    concatenated_frames[0].save(
        output_path,
        save_all=True,
        append_images=concatenated_frames[1:],
        duration=durations[0],  # Set the duration of the GIF
        loop=0
    )

    return output_path


def generate_random_line_trajectories(
    batch_size: int,
    n: int,
    t: int,
    grid_size: int
):
    # Generate random start and end points for each batch element to define line directions
    p1 = np.random.randint(0, grid_size, size=(
        batch_size, 2))  # (batch_size, 2)
    p2 = np.random.randint(0, grid_size, size=(
        batch_size, 2))  # (batch_size, 2)

    # Calculate direction vectors and normalize
    directions = p2 - p1  # (batch_size, 2)
    norms = np.linalg.norm(
        directions, axis=1, keepdims=True)  # (batch_size, 1)
    # Normalize directions to unit vectors (batch_size, 2)
    directions = directions / norms

    # Generate random starting points anywhere in the grid for each batch and point
    points = np.random.randint(0, grid_size, size=(
        batch_size, n, 2))  # (batch_size, n, 2)

    # Prepare the direction vector for broadcasting across timesteps
    directions_expanded = directions[:, np.newaxis, :]  # (batch_size, 1, 2)

    # Create the trajectory array to store points at each timestep
    # (timesteps, batch_size, n, 2)
    trajectories = np.zeros((t + 1, batch_size, n, 2))

    # Set the initial points
    trajectories[0] = points  # (batch_size, n, 2)

    # Move the points for each timestep
    for step in range(1, t + 1):
        trajectories[step] = trajectories[step - 1] + \
            directions_expanded  # Move points along the line direction

    # Reshape the array to be (batch_size, num_points, timesteps, 2)
    # (batch_size, num_points, timesteps, 2)
    trajectories = np.transpose(trajectories, (1, 2, 0, 3))

    return trajectories
