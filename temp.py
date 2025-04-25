    # def update_command_curriculum(self, env_ids):
    #     """ Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # If the tracking reward is above 80% of the maximum, increase the range of commands
    #     if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_lin_vel"]:
    #         self.command_x_range[0] = np.clip(self.command_x_range[0] - 0.2, -self.limit_vel_x[0], 0.).item()
    #         self.command_x_range[1] = np.clip(self.command_x_range[1] + 0.2, 0., self.limit_vel_x[1]).item()

    #         # Increase the range of commands for y
    #         self.command_y_range[0] = np.clip(self.command_y_range[0] - 0.2, -self.limit_vel_y[0], 0.).item()
    #         self.command_y_range[1] = np.clip(self.command_y_range[1] + 0.2, 0., self.limit_vel_y[1]).item()
        
    #     if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_ang_vel"]:
    #     # Increase the range of commands for yaw
    #         self.command_yaw_range[0] = np.clip(self.command_yaw_range[0] - 0.2, -self.limit_vel_yaw[0], 0.).item()
    #         self.command_yaw_range[1] = np.clip(self.command_yaw_range[1] + 0.2, 0., self.limit_vel_yaw[1]).item()
    


    def _visualize_terrain_heights(self):
        """
        Spawns (or updates) a PointInstancer of small spheres for every cell in self.height_samples,
        but only for the main terrain region (excluding the border).
        """
        stage = self._stage  # or get_current_stage()

        # 1) Create a dedicated Scope for the debug instancer
        parent_scope_path = "/World/DebugTerrainHeights"
        parent_scope_prim = stage.GetPrimAtPath(parent_scope_path)
        if not parent_scope_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=parent_scope_path,
                attributes={}
            )

        # 2) Construct a PointInstancer prim if not already there
        point_instancer_path = f"{parent_scope_path}/terrain_points_instancer"
        point_instancer_prim = stage.GetPrimAtPath(point_instancer_path)
        if not point_instancer_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=point_instancer_path,
                attributes={}
            )

        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(point_instancer_path))

        # 3) Create/ensure we have a single prototype (Sphere) under the PointInstancer
        prototype_index = 0
        proto_path = f"{point_instancer_path}/prototype_Sphere"
        prototype_prim = stage.GetPrimAtPath(proto_path)
        if not prototype_prim.IsValid():
            # Create a sphere prototype
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=proto_path,
                attributes={"radius": 0.02},  # adjust sphere size as you wish
            )
        # This step ensures the point-instancer references the prototype as well
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(proto_path)

        # 4) Build up the positions (and protoIndices) for each cell of the *main* height field
        tot_rows = self.terrain.tot_rows   # i dimension
        tot_cols = self.terrain.tot_cols   # j dimension
        border   = self.terrain.border     # integer # of “cells” that define the border thickness

        positions = []
        proto_indices = []

        # Only iterate within the interior region [border, (tot_rows - border)) and [border, (tot_cols - border))
        for i in range(border, tot_rows - border):
            for j in range(border, tot_cols - border):
                # Convert row/col -> world coordinates
                px = i * self.terrain.horizontal_scale - self.terrain.border_size
                py = j * self.terrain.horizontal_scale - self.terrain.border_size
                pz = float(self.height_samples[i, j] * self.terrain.vertical_scale)

                positions.append(Gf.Vec3f(px, py, pz))
                proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign the arrays to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # Optionally give these debug spheres a color by modifying the prototype itself:
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 1.0)])


    def _visualize_depression_indices(self):
        """
        Creates (or updates) a PointInstancer of small spheres at z=0 
        for each (x,y) entry
        """

        stage = self._stage  # Or get_current_stage()

        # 1) Create a dedicated Scope for debugging these indices
        debug_scope_path = "/World/DebugDepressionIndices"
        if not stage.GetPrimAtPath(debug_scope_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=debug_scope_path,
                attributes={}
            )

        # 2) Create a PointInstancer for all depression indices
        instancer_path = f"{debug_scope_path}/DepressionIndicesPointInstancer"
        if not stage.GetPrimAtPath(instancer_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=instancer_path,
                attributes={}
            )
        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))

        # 3) Make sure there's a prototype sphere
        sphere_proto_path = f"{instancer_path}/DepressionIndexSphere"
        if not stage.GetPrimAtPath(sphere_proto_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=sphere_proto_path,
                attributes={"radius": 0.02},  # adjust size as desired
            )
        # Ensure the instancer references the prototype
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(sphere_proto_path)

        # 4) Collect positions and prototype indices
        positions = []
        proto_indices = []

        prototype_index = 0  # single prototype

        for terrain_entry in self.terrain_details:
            terrain_name = terrain_entry[15]
            if terrain_name == "central_depression_terrain":
                bx_start = int(terrain_entry[10])
                bx_end   = int(terrain_entry[11])
                by_start = int(terrain_entry[12])
                by_end   = int(terrain_entry[13])

                # For each (i, j) in that rectangle
                for i in range(bx_start, bx_end):
                    for j in range(by_start, by_end):
                        # Convert heightfield indices to world coordinates
                        px = i * self.terrain.horizontal_scale - self.terrain.border_size
                        py = j * self.terrain.horizontal_scale - self.terrain.border_size
                        pz = 0.0  # place at ground (z=0)
                        positions.append(Gf.Vec3f(px, py, pz))
                        proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # (Optional) Color the debug spheres differently
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])  # green for clarity



















            # Iterate by index instead of zip
            for idx in range(px.shape[0]):
                i = int(px[idx].item())
                j = int(py[idx].item())
                # Convert (i,j) indices into world coordinates.
                # Here we assume that each cell spans a distance equal to terrain.horizontal_scale,
                # and that the terrain's origin offset is given by terrain.border_size.
                cell_x_min = (i-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_x_max = (i + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_min = (j-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_max = (j + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_x = i * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y = j * self.terrain.horizontal_scale - self.terrain.border_size

                # For each cell defined by cell_x_min, cell_x_max, cell_y_min, cell_y_max:
                mask = (
                    (particle_positions_np[:, 0] >= cell_x_min) &
                    (particle_positions_np[:, 0] < cell_x_max) &
                    (particle_positions_np[:, 1] >= cell_y_min) &
                    (particle_positions_np[:, 1] < cell_y_max)
                )
                if np.any(mask):
                    top_z = float(np.max(particle_positions_np[mask, 2]))
                    top_z = min(top_z, 1*self.terrain.vertical_scale)
                    self.height_samples[i, j] = top_z 
                else:
                    top_z = self.height_samples[i, j] * self.terrain.vertical_scale
                if visualize:
                    positions.append(Gf.Vec3f(cell_x, cell_y, float(top_z)))
                    proto_indices.append(prototype_index)