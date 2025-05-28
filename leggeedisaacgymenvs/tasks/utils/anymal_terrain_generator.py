import math

import numpy as np
import torch
from leggeedisaacgymenvs.utils.terrain_utils.terrain_utils import *


# terrain generator

class Terrain:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 30

        # Map dimensions in meters
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        terrain_types = self.cfg["terrain_types"]

        if cfg["flat"]:
            self.env_rows = cfg["numLevels"]
            self.env_cols = cfg["numTerrains"]
        else:
            self.env_rows = sum(tt.get("row_count", 1) for tt in terrain_types)
            self.env_cols = max(t["col_count"] for t in terrain_types)  # total columns = max of 'count' across all types
        
        self.num_maps = self.env_rows * self.env_cols
        # Each sub-rectangle (sub-terrain) dimensions in "heightfield" pixels
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        # Master heightfield storage
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # We'll keep track of each sub-terrain's info in a list:
        self.terrain_details = []

        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        # Actually build the terrain
        if not cfg["flat"]:
            self.deformable_curriculum()
        else:
            self.full_flat_terrain()
        
        self.heightsamples = self.height_field_raw

        # Convert to tri-mesh
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"]
        )

    def deformable_curriculum(self):
        """
        Create sub-terrains in a deterministic 'in-order' fashion based on
        the `terrain_types` array from the config, repeating as needed.
        """

        # All possible terrain type definitions from config:
        terrain_type_list = self.cfg["terrain_types"]  # e.g. a list of dicts
        current_row = 0
        for terrain_type_info in terrain_type_list:
            name = str(terrain_type_info["name"])
            row_count = terrain_type_info["row_count"]  # Number of rows of this type
            col_count = terrain_type_info["col_count"]  # Number of terrains of this type
            level = terrain_type_info["level"]
            size = terrain_type_info.get("size", 0.0)
            depth = terrain_type_info.get("depth", 0.0)
            system = terrain_type_info.get("system", 0)
            particles = int(terrain_type_info.get("particle_present", "False"))
            compliant = int(terrain_type_info.get("compliant", "False"))
            fluid = bool(
                self.cfg["particles"]
                    .get(f"system{system}", {})
                    .get("particle_grid_fluid", False)
            )
            for r in range(row_count):
                for c in range(col_count):                
                    idx = len(self.terrain_details)  # Unique terrain index
                    terrain = SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale,
                    )

                    # Assign terrain heightfield based on type
                    if name == "flat":
                        flat_terrain(terrain, height_meters=0.0)
                    elif name == "rough":
                        random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)                
                    elif name == "central_depression_terrain":

                        central_depression_terrain(
                            terrain, depression_depth=-abs(depth), platform_height=0.0, depression_size=size
                        )
                    else:
                        flat_terrain(terrain, height_meters=0.0)

                    # Compute terrain placement in row i, col j
                    start_x = self.border + (current_row + r) * self.length_per_env_pixels
                    end_x = start_x + self.length_per_env_pixels
                    start_y = self.border + c * self.width_per_env_pixels
                    end_y = start_y + self.width_per_env_pixels

                    self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw


                    # bounding region in HF indices:
                    if name == "central_depression_terrain":
                        lx0 = start_x + terrain.depression_indices["start_x"]
                        lx1 = start_x + terrain.depression_indices["end_x"]
                        ly0 = start_y + terrain.depression_indices["start_y"]
                        ly1 = start_y + terrain.depression_indices["end_y"]
                    else:
                        lx0, lx1, ly0, ly1 = (start_x, end_x, start_y, end_y)

                    # Store the origin of the terrain
                    env_origin_x = ((current_row + r) + 0.5) * self.env_length
                    env_origin_y = (c + 0.5) * self.env_width
                    center_x1 = int((self.env_length / 2 - 1) / self.horizontal_scale)
                    center_x2 = int((self.env_length / 2 + 1) / self.horizontal_scale)
                    center_y1 = int((self.env_width / 2 - 1) / self.horizontal_scale)
                    center_y2 = int((self.env_width / 2 + 1) / self.horizontal_scale)
                    env_origin_z = np.max(
                        terrain.height_field_raw[center_x1:center_x2, center_y1:center_y2]
                    ) * self.vertical_scale 
                    self.env_origins[current_row + r, c] = [env_origin_x, env_origin_y, env_origin_z]
                    terrain_type = self.infer_terrain_type(compliant, particles, fluid)

                    # Store terrain details
                    self.terrain_details.append((
                        idx,
                        level,
                        current_row+r,
                        c,
                        terrain_type,
                        particles,
                        compliant,
                        system,
                        depth,
                        size,
                        lx0,
                        lx1,
                        ly0,
                        ly1,
                        fluid,
                    ))
            current_row += row_count

    @staticmethod
    def infer_terrain_type(compliant: int, particles: int, fluid: bool) -> int:
        """
        Infer terrain type index based on compliance and particle flags:
          0: rigid    (non-compliant, no particles)
          1: compliant (compliant, no particles)
          2: granular (particles, non-fluid)
          3: fluid    (particles and fluid)
        """
        if particles:
            return 3 if fluid else 2
        else:
            return 1 if compliant else 0
        
    def full_flat_terrain(self, height_meters=0.0):
            """
            Generate flat terrain for all sub-terrains instead of random obstacles.
            """
            for k in range(self.num_maps):
                # Env coordinates in the world
                (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

                # Heightfield coordinate system from now on
                start_x = self.border + i * self.length_per_env_pixels
                end_x   = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y   = self.border + (j + 1) * self.width_per_env_pixels

                # Create a SubTerrain for this environment
                terrain = SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )

                # Call the flat_terrain function from terrain_utils
                flat_terrain(terrain, height_meters=height_meters)

                # Copy the new flat terrain into our global height_field_raw
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                # Compute the average origin height for placing robots
                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                
                # For a flat terrain, the terrain is uniform, but let's still compute
                x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
                y1 = int((self.env_width  / 2.0 - 1) / self.horizontal_scale)
                y2 = int((self.env_width  / 2.0 + 1) / self.horizontal_scale)
                
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
                self.terrain_details.append((
                    k,         # Unique terrain index
                    0,         # level (default)
                    i,         # row index
                    j,         # column index
                    0,
                    0,         # particles (default)
                    0,         # compliant (default)
                    0,         # system (default)
                    0,         # depth (default)
                    0,         # size (default)
                    start_x,   # lx0: start index in x
                    end_x,     # lx1: end index in x
                    start_y,   # ly0: start index in y
                    end_y,     # ly1: end index in y
                    0
                ))