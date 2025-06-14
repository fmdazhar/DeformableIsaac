import copy

import carb
import numpy as np
import omni.usd
import torch
from omni.isaac.core.utils.extensions import enable_extension
from leggeedisaacgymenvs.utils.config_utils.default_scene_params import *


class SimConfig:
    def __init__(self, config: dict = None):
        if config is None:
            config = dict()

        self._config = config
        self._cfg = config.get("task", dict())
        self._parse_config()

        if self._config["test"] == True:
            self._sim_params["enable_scene_query_support"] = True

        if (
            self._config["headless"] == True
            and not self._sim_params["enable_cameras"]
            and not self._config["enable_livestream"]
            and not self._config.get("enable_recording", False)
        ):
            self._sim_params["use_fabric"] = False
            self._sim_params["enable_viewport"] = False
        else:
            self._sim_params["enable_viewport"] = True
            enable_extension("omni.kit.viewport.bundle")
            if self._sim_params["enable_cameras"] or self._config.get("enable_recording", False):
                enable_extension("omni.replicator.isaac")

        self._sim_params["sim_device"] = self._config["sim_device"]

        self._adjust_dt()

        if self._sim_params["disable_contact_processing"]:
            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", True)

        carb.settings.get_settings().set_bool("/physics/physxDispatcher", True)
        # Force the background grid off all the time for RL tasks, to avoid the grid showing up in any RL camera task
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
        # Disable framerate limiting which might cause rendering slowdowns
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)
        # enable scene graph instancing
        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

        # # App
        # carb.settings.get_settings().set("/app/enableDeveloperWarnings", False)
        # carb.settings.get_settings().set("/app/scripting/ignoreWarningDialog", True)

        # # UI Console
        # carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/verbose", False)
        # carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/info", False)
        # carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/warning", False)
        # carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/error", False)
        # carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/fatal", False)

        # # Log 
        # carb.settings.get_settings().set("/log/debugConsoleLevel", "Fatal")  # verbose"|"info"|"warning"|"error"|"fatal"
        # carb.settings.get_settings().set("/log/enabled", False)
        # carb.settings.get_settings().set("/log/outputStreamLevel", "Error")
        # carb.settings.get_settings().set("/log/fileLogLevel", "Error")

        import omni.ui 
        # Dock floating UIs this might not be needed anymore as extensions dock themselves
        # Method for docking a particular window to a location
        def dock_window(space, name, location, ratio=0.5):
            window = omni.ui.Workspace.get_window(name)
            if window and space:
                window.dock_in(space, location, ratio=ratio)
            return window
        # Acquire the main docking station
        main_dockspace = omni.ui.Workspace.get_window("DockSpace")
        dock_window(main_dockspace, "Content", omni.ui.DockPosition.BOTTOM, 0.3)

        window = omni.ui.Workspace.get_window("Content")
        if window:
            window.visible = False
        window = omni.ui.Workspace.get_window("Simulation Settings")
        if window:
            window.visible = False

        # workaround for asset root search hang
        carb.settings.get_settings().set_string(
            "/persistent/isaac/asset_root/default",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0",
        )
        carb.settings.get_settings().set_string(
            "/persistent/isaac/asset_root/nvidia",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0",
        )

        # make sure the correct USD update flags are set
        if self._sim_params["use_fabric"]:
            carb.settings.get_settings().set_bool("/physics/updateToUsd", False)
            carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", False)
            carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd", False)
            carb.settings.get_settings().set_bool("/physics/updateForceSensorsToUsd", False)
            carb.settings.get_settings().set_bool("/physics/outputVelocitiesLocalSpace", False)
            carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
            carb.settings.get_settings().set_bool("/physics/fabricUpdateVelocities", False)
            carb.settings.get_settings().set_bool("/physics/fabricUpdateForceSensors", False)
            carb.settings.get_settings().set_bool("/physics/fabricUpdateJointStates", False)

    def _parse_config(self):
        # general sim parameter
        self._sim_params = copy.deepcopy(default_sim_params)
        self._default_physics_material = copy.deepcopy(default_physics_material)
        sim_cfg = self._cfg.get("sim", None)
        if sim_cfg is not None:
            for opt in sim_cfg.keys():
                if opt in self._sim_params:
                    if opt == "default_physics_material":
                        for material_opt in sim_cfg[opt]:
                            self._default_physics_material[material_opt] = sim_cfg[opt][material_opt]
                    else:
                        self._sim_params[opt] = sim_cfg[opt]
                else:
                    print("Sim params does not have attribute: ", opt)
        self._sim_params["default_physics_material"] = self._default_physics_material

        # physx parameters
        self._physx_params = copy.deepcopy(default_physx_params)
        if sim_cfg is not None and "physx" in sim_cfg:
            for opt in sim_cfg["physx"].keys():
                if opt in self._physx_params:
                    self._physx_params[opt] = sim_cfg["physx"][opt]
                else:
                    print("Physx sim params does not have attribute: ", opt)

        self._sanitize_device()

    def _sanitize_device(self):
        if self._sim_params["use_gpu_pipeline"]:
            self._physx_params["use_gpu"] = True

        # device should be in sync with pipeline
        if self._sim_params["use_gpu_pipeline"]:
            self._config["sim_device"] = f"cuda:{self._config['device_id']}"
        else:
            self._config["sim_device"] = "cpu"

        # also write to physics params for setting sim device
        self._physx_params["sim_device"] = self._config["sim_device"]

        print("Pipeline: ", "GPU" if self._sim_params["use_gpu_pipeline"] else "CPU")
        print("Pipeline Device: ", self._config["sim_device"])
        print("Sim Device: ", "GPU" if self._physx_params["use_gpu"] else "CPU")

    def parse_actor_config(self, actor_name):
        actor_params = copy.deepcopy(default_actor_options)
        if "sim" in self._cfg and actor_name in self._cfg["sim"]:
            actor_cfg = self._cfg["sim"][actor_name]
            for opt in actor_cfg.keys():
                if actor_cfg[opt] != -1 and opt in actor_params:
                    actor_params[opt] = actor_cfg[opt]
                elif opt not in actor_params:
                    print("Actor params does not have attribute: ", opt)

        return actor_params

    def _get_actor_config_value(self, actor_name, attribute_name, attribute=None):
        actor_params = self.parse_actor_config(actor_name)

        if attribute is not None:
            if attribute_name not in actor_params:
                return attribute.Get()

            if actor_params[attribute_name] != -1:
                return actor_params[attribute_name]
            elif actor_params["override_usd_defaults"] and not attribute.IsAuthored():
                return self._physx_params[attribute_name]
        else:
            if actor_params[attribute_name] != -1:
                return actor_params[attribute_name]
            
    def _adjust_dt(self):
        # re-evaluate rendering dt to simulate physics substeps
        physics_dt = self.sim_params["dt"]
        rendering_dt = self.sim_params["rendering_dt"]

        # by default, rendering dt = physics dt
        if rendering_dt <= 0:
            rendering_dt = physics_dt

        self.task_config["renderingInterval"] = max(round((1/physics_dt) / (1/rendering_dt)), 1)

        self.sim_params["rendering_dt"] = physics_dt

    @property
    def sim_params(self):
        return self._sim_params

    @property
    def config(self):
        return self._config

    @property
    def task_config(self):
        return self._cfg

    @property
    def physx_params(self):
        return self._physx_params

    def get_physics_params(self):
        return {**self.sim_params, **self.physx_params}

    def _get_physx_collision_api(self, prim):
        from pxr import PhysxSchema, UsdPhysics

        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        return physx_collision_api

    def _get_physx_rigid_body_api(self, prim):
        from pxr import PhysxSchema, UsdPhysics

        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rb_api:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        return physx_rb_api

    def _get_physx_articulation_api(self, prim):
        from pxr import PhysxSchema, UsdPhysics

        arti_api = PhysxSchema.PhysxArticulationAPI(prim)
        if not arti_api:
            arti_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        return arti_api

    def set_contact_offset(self, name, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        contact_offset = physx_collision_api.GetContactOffsetAttr()
        # if not contact_offset:
        #     contact_offset = physx_collision_api.CreateContactOffsetAttr()
        if value is None:
            value = self._get_actor_config_value(name, "contact_offset", contact_offset)
        if value != -1:
            contact_offset.Set(value)

    def set_rest_offset(self, name, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        rest_offset = physx_collision_api.GetRestOffsetAttr()
        # if not rest_offset:
        #     rest_offset = physx_collision_api.CreateRestOffsetAttr()
        if value is None:
            value = self._get_actor_config_value(name, "rest_offset", rest_offset)
        if value != -1:
            rest_offset.Set(value)

    def set_position_iteration(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_position_iteration_count = physx_rb_api.GetSolverPositionIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_position_iteration_count", solver_position_iteration_count
            )
        if value != -1:
            solver_position_iteration_count.Set(value)

    def set_velocity_iteration(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_velocity_iteration_count = physx_rb_api.GetSolverVelocityIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_velocity_iteration_count", solver_velocity_iteration_count
            )
        if value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_max_depenetration_velocity(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        max_depenetration_velocity = physx_rb_api.GetMaxDepenetrationVelocityAttr()
        if value is None:
            value = self._get_actor_config_value(name, "max_depenetration_velocity", max_depenetration_velocity)
        if value != -1:
            max_depenetration_velocity.Set(value)

    def set_sleep_threshold(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        sleep_threshold = physx_rb_api.GetSleepThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
        if value != -1:
            sleep_threshold.Set(value)

    def set_stabilization_threshold(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        stabilization_threshold = physx_rb_api.GetStabilizationThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
        if value != -1:
            stabilization_threshold.Set(value)

    def set_gyroscopic_forces(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        enable_gyroscopic_forces = physx_rb_api.GetEnableGyroscopicForcesAttr()
        if value is None:
            value = self._get_actor_config_value(name, "enable_gyroscopic_forces", enable_gyroscopic_forces)
        if value != -1:
            enable_gyroscopic_forces.Set(value)

    def set_density(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        density = physx_rb_api.GetDensityAttr()
        if value is None:
            value = self._get_actor_config_value(name, "density", density)
        if value != -1:
            density.Set(value)
            # auto-compute mass
            self.set_mass(prim, 0.0)

    def set_mass(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        mass = physx_rb_api.GetMassAttr()
        if value is None:
            value = self._get_actor_config_value(name, "mass", mass)
        if value != -1:
            mass.Set(value)

    def retain_acceleration(self, prim):
        # retain accelerations if running with more than one substep
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        if self._sim_params["substeps"] > 1:
            physx_rb_api.GetRetainAccelerationsAttr().Set(True)

    def make_kinematic(self, name, prim, cfg, value=None):
        # make rigid body kinematic (fixed base and no collision)
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if value is None:
            value = self._get_actor_config_value(name, "make_kinematic")
        if value == True:
            # parse through all children prims
            prims = [prim]
            while len(prims) > 0:
                cur_prim = prims.pop(0)
                rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())

                if rb:
                    rb.CreateKinematicEnabledAttr().Set(True)

                children_prims = cur_prim.GetPrim().GetChildren()
                prims = prims + children_prims

    def set_articulation_position_iteration(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        solver_position_iteration_count = arti_api.GetSolverPositionIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_position_iteration_count", solver_position_iteration_count
            )
        if value != -1:
            solver_position_iteration_count.Set(value)

    def set_articulation_velocity_iteration(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        solver_velocity_iteration_count = arti_api.GetSolverVelocityIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_velocity_iteration_count", solver_velocity_iteration_count
            )
        if value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_articulation_sleep_threshold(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        sleep_threshold = arti_api.GetSleepThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
        if value != -1:
            sleep_threshold.Set(value)

    def set_articulation_stabilization_threshold(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        stabilization_threshold = arti_api.GetStabilizationThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
        if value != -1:
            stabilization_threshold.Set(value)

    def apply_rigid_body_settings(self, name, prim, cfg, is_articulation):
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        rb_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
        if not physx_rb_api:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # if it's a body in an articulation, it's handled at articulation root
        if not is_articulation:
            self.make_kinematic(name, prim, cfg, cfg["make_kinematic"])
        self.set_position_iteration(name, prim, cfg["solver_position_iteration_count"])
        self.set_velocity_iteration(name, prim, cfg["solver_velocity_iteration_count"])
        self.set_max_depenetration_velocity(name, prim, cfg["max_depenetration_velocity"])
        self.set_sleep_threshold(name, prim, cfg["sleep_threshold"])
        self.set_stabilization_threshold(name, prim, cfg["stabilization_threshold"])
        self.set_gyroscopic_forces(name, prim, cfg["enable_gyroscopic_forces"])

        # density and mass
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if mass_api is None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_attr = mass_api.GetMassAttr()
        density_attr = mass_api.GetDensityAttr()
        if not mass_attr:
            mass_attr = mass_api.CreateMassAttr()
        if not density_attr:
            density_attr = mass_api.CreateDensityAttr()

        if cfg["density"] != -1:
            density_attr.Set(cfg["density"])
            mass_attr.Set(0.0)  # mass is to be computed
        elif cfg["override_usd_defaults"] and not density_attr.IsAuthored() and not mass_attr.IsAuthored():
            density_attr.Set(self._physx_params["density"])

        self.retain_acceleration(prim)

    def apply_rigid_shape_settings(self, name, prim, cfg):
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()

        # collision APIs
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

        self.set_contact_offset(name, prim, cfg["contact_offset"])
        self.set_rest_offset(name, prim, cfg["rest_offset"])

    def apply_articulation_settings(self, name, prim, cfg):
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()

        is_articulation = False
        # check if is articulation
        prims = [prim]
        while len(prims) > 0:
            prim_tmp = prims.pop(0)
            articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, prim_tmp.GetPath())
            physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim_tmp.GetPath())

            if articulation_api or physx_articulation_api:
                is_articulation = True

            children_prims = prim_tmp.GetPrim().GetChildren()
            prims = prims + children_prims

        # parse through all children prims
        prims = [prim]
        while len(prims) > 0:
            cur_prim = prims.pop(0)
            rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())
            collision_body = UsdPhysics.CollisionAPI.Get(stage, cur_prim.GetPath())
            articulation = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
            if rb:
                self.apply_rigid_body_settings(name, cur_prim, cfg, is_articulation)
            if collision_body:
                self.apply_rigid_shape_settings(name, cur_prim, cfg)

            if articulation:
                articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
                physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, cur_prim.GetPath())

                # enable self collisions
                enable_self_collisions = physx_articulation_api.GetEnabledSelfCollisionsAttr()
                if cfg["enable_self_collisions"] != -1:
                    enable_self_collisions.Set(cfg["enable_self_collisions"])

                self.set_articulation_position_iteration(name, cur_prim, cfg["solver_position_iteration_count"])
                self.set_articulation_velocity_iteration(name, cur_prim, cfg["solver_velocity_iteration_count"])
                self.set_articulation_sleep_threshold(name, cur_prim, cfg["sleep_threshold"])
                self.set_articulation_stabilization_threshold(name, cur_prim, cfg["stabilization_threshold"])

            children_prims = cur_prim.GetPrim().GetChildren()
            prims = prims + children_prims
