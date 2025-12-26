import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
np.random.seed(42)
import csv
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_to_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from src.follower import GreedyGeodesicFollower

import ast
import random

from magnum import Vector3


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
        
    with open(cfg.on_data_path) as f:        
        data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    logging.info(f"Loaded {len(data)} questions.")

    # Load VLM
    vlm = VLM(cfg.vlm)

    num = 0
    cnt_data = 0
    results_all = []
    success_all = 0
    spl_all = 0
    success_bed = 0
    spl_bed = 0
    num_bed = 0
    success_toilet = 0
    spl_toilet = 0
    num_toilet = 0
    success_tv_monitor = 0
    spl_tv_monitor = 0
    num_tv_monitor = 0
    success_chair = 0
    spl_chair = 0
    num_chair = 0
    success_sofa = 0
    spl_sofa = 0
    num_sofa = 0
    success_plant = 0
    spl_plant = 0
    num_plant = 0

    for ind in tqdm(range(len(data))): #####################################iter
        mf = 0
        num += 1
        flag = 0
        # Init
        sin_data = data[ind]
        scene = sin_data["scene"]
        goal = sin_data["goal"]
        starting_distance = float(sin_data["starting_distance"])
        goal_loc = np.array(ast.literal_eval(sin_data["goal_loc"]))
        init_pts = [float(sin_data["init_x"]), float(sin_data["init_y"]), float(sin_data["init_z"])]
        init_angle = quat_to_angle_axis(quaternion.from_float_array(np.array(ast.literal_eval(sin_data["init_angle"]))[[3, 0, 1, 2]]))[0] ##############################wait
        logging.info(f"\n========\nIndex: {ind} Scene: {scene} Goal: {goal} init_angle: {init_angle}")


        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(ind))
        # os.makedirs(episode_data_dir, exist_ok=True)
        result = {"ind": ind}

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = os.path.join(
            # cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
            cfg.scene_data_path, scene + ".glb"
        )
        navmesh_file = os.path.join(
            # cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
            cfg.scene_data_path, scene + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        # num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio) 
        num_step = 500 ##########################
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        # pts = [7.03487, 2.06447, 2.26951]
        agent_state.position = Vector3(*pts)
        agent_state.rotation = quat_from_angle_axis(angle, np.array([0, 1, 0]))
        agent.set_state(agent_state)
        follower = GreedyGeodesicFollower(pathfinder = pathfinder, agent = agent)

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        # for cnt_step in range(num_step):
        cnt_step = 0
        while True:
            if cnt_step>=num_step:
                break
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            pts_normal = pos_habitat_to_normal(pts)
            result[step_name] = {"pts": pts, "angle": angle}

            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            if cfg.save_obs:
                 plt.imsave(
                     os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                 )
            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:

                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )


                # Get VLM relevancy
                prompt_rel = f"\nConsider the goal: '{goal}'. Are you confident about find the goal in the current view?"
                # logging.info(f"Prompt Rel: {prompt_text}")
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                logging.info(f"Rel - Prob: {smx_vlm_rel}")

                if smx_vlm_rel[0]>=0.75:
                    flag = 1
                    # rgb_im.save(
                    #     os.path.join(episode_data_dir, "final.png")
                    # )
                    prompt_final = f"\nConsider the goal: '{goal}', where is it in img? Answer with a image coordinate!(do not answer like 'in the center of')"
                    location = vlm.generate(prompt_final, rgb_im)
                    # print(f"prompt: {location}")
                    width, height = rgb_im.size
                    try:
                        goal_pix = ast.literal_eval(location)
                        if len(goal_pix)>3:
                            goal_pix = [(min(goal_pix[0], goal_pix[2])+abs(goal_pix[0]-goal_pix[2])/2)*width, (min(goal_pix[1], goal_pix[3])+abs(goal_pix[1]-goal_pix[3])/2)*height]
                            # check check and save the img
                            rgb_im_final_draw = rgb_im.copy()                  
                            draw_final = ImageDraw.Draw(rgb_im_final_draw)
                            draw_final.ellipse(
                                (
                                    goal_pix[0] - cfg.visual_prompt.circle_radius,
                                    goal_pix[1] - cfg.visual_prompt.circle_radius,
                                    goal_pix[0] + cfg.visual_prompt.circle_radius,
                                    goal_pix[1] + cfg.visual_prompt.circle_radius,
                                ),
                                fill=(200, 200, 200, 255),
                                outline=(0, 0, 0, 255),
                                width=3,
                            )
                            draw_final.text(
                                # tuple(goal_pix.astype(int).tolist()),
                                tuple(goal_pix),
                                'G',
                                font=fnt,
                                fill=(0, 0, 0, 255),
                                anchor="mm",
                                font_size=12,
                            )

                    except Exception as e:
                        logging.error(f"save error: {e}")
                        logging.info(location)

                    target_pos_3d = get_3d_goal_from_pixel(
                        goal_pix[0], goal_pix[1], depth, cam_pose, cam_intr
                    )
                    logging.info(f"Target World 3D: {target_pos_3d}")

                    approach_dist, _ = follower.find_path(Vector3(*target_pos_3d), current_angle)
                    distance_traveled += approach_dist
                    cnt_step += int(approach_dist)
                    final_pos = agent.get_state().position
                    dist_to_obj = np.linalg.norm(np.array(final_pos) - np.array(target_pos_3d))
                    
                    logging.info(f"Final Distance to Object: {dist_to_obj:.4f}m")
                    
                    if dist_to_obj < 0.15:
                        success = 1
                    if mf == 0:
                        spl = 1
                    else:
                        spl = min(starting_distance/(mf*0.25), 1.0)

                    ## Episode summary
                    logging.info(f"\n== Episode Summary")
                    logging.info(f"Scene: {scene}, Goal: {goal}")
                    logging.info(f"Success : {success}, SPL : {spl}")
                    # total
                    if goal=="bed":
                        success_bed +=success
                        spl_bed +=spl
                        num_bed +=1
                    elif goal=="toilet":
                        success_toilet +=success
                        spl_toilet +=spl
                        num_toilet +=1
                    elif goal=="tv_monitor":
                        success_tv_monitor +=success
                        spl_tv_monitor +=spl
                        num_tv_monitor +=1
                    elif goal=="chair":
                        success_chair +=success
                        spl_chair +=spl
                        num_chair +=1
                    elif goal=="sofa":
                        success_sofa +=success
                        spl_sofa +=spl
                        num_sofa +=1
                    elif goal=="plant":
                        success_plant +=success
                        spl_plant +=spl
                        num_plant +=1
                    success_all +=success
                    spl_all += spl
                    logging.info(f"Success_All : {success_all/num}, SPL_All : {spl_all/num}")
                    logging.info(f"== Class Summary ==")
                    try:
                        logging.info(f"bed: Success: {success_bed/num_bed} SPL: {spl_bed/num_bed}")
                        logging.info(f"toilet: Success: {success_toilet/num_toilet} SPL: {spl_toilet/num_toilet}")
                        logging.info(f"tv_monitor: Success: {success_tv_monitor/num_tv_monitor} SPL: {spl_tv_monitor/num_tv_monitor}")
                        logging.info(f"chair: Success: {success_chair/num_chair} SPL: {spl_chair/num_chair}")
                        logging.info(f"sofa: Success: {success_sofa/num_sofa} SPL: {spl_sofa/num_sofa}")
                        logging.info(f"plant: Success: {success_plant/num_plant} SPL: {spl_plant/num_plant}")
                    except Exception as e:
                        logging.error(f"e: {e}")
         
                    break




                # Get frontier candidates
                prompt_points_pix = []
                if cfg.use_active:
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            img_width,
                            img_height,
                            cam_intr,
                            cam_pose_tsdf,
                            **cfg.visual_prompt,
                        )
                    )
                    fig.tight_layout()
                    # plt.savefig(
                    #     os.path.join(
                    #         episode_data_dir, "{}_prompt_points.png".format(cnt_step)
                    #     )
                    # )
                    plt.close()

                # Visual prompting
                draw_letters = ["A", "B", "C", "D"]  # always four
                fnt = ImageFont.truetype(
                    "data/Open_Sans/static/OpenSans-Regular.ttf",
                    30,
                )
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                    rgb_im_draw = rgb_im.copy()
                    draw = ImageDraw.Draw(rgb_im_draw)
                    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                        draw.ellipse(
                            (
                                point_pix[0] - cfg.visual_prompt.circle_radius,
                                point_pix[1] - cfg.visual_prompt.circle_radius,
                                point_pix[0] + cfg.visual_prompt.circle_radius,
                                point_pix[1] + cfg.visual_prompt.circle_radius,
                            ),
                            fill=(200, 200, 200, 255),
                            outline=(0, 0, 0, 255),
                            width=3,
                        )
                        draw.text(
                            tuple(point_pix.astype(int).tolist()),
                            draw_letters[prompt_point_ind],
                            font=fnt,
                            fill=(0, 0, 0, 255),
                            anchor="mm",
                            font_size=12,
                        )
                    # rgb_im_draw.save(
                    #     os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
                    # )

                    
                    sv = np.array([])
                    for index in range(actual_num_prompt_points):
                        prompt_u = f"\nYou are an assistant who are exploring the envrionment to search the goal: '{goal}'.The image is your first point of view.\nDo you think location {draw_letters[index]} (black letter on the image) is valuable to find the goal?"
                        _sv = vlm.get_loss(rgb_im_draw, prompt_u, ["Yes", "No"])[0]
                        # _sv = math.sqrt(_sv)
                        sv = np.append(sv, _sv)
                    logging.info(f"sv: {sv}")
                    tsdf_planner.integrate_sem(
                        # sem_pix=sv-(1-smx_vlm_rel[0]),
                        sem_pix=sv,
                        radius=1.5, ##################################
                        obs_weight=1.5,
                    )
                    if cnt_step>0:
                        try:
                            tsdf_planner.integrate_negative_sem(
                                p=max_normal,
                                sem=-(1-smx_vlm_rel[0]),
                                radius=2.0,
                                obs_weight=1.0,
                            )
                        except NameError:
                            logging.info("max_normal name error")
                            
                result[step_name]["smx_vlm_rel"] = smx_vlm_rel
            else:
                logging.info("Skipping black image!")
                # result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])

            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, pts_pix, fig, max_normal = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                max_normal = np.append(max_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()
                # plt.savefig(
                #     os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
                # )
                plt.close()

                step_goal_loc = Vector3(*pts)
                step_len, mfc = follower.find_path(step_goal_loc, angle)
                mf += mfc
                logging.info(step_len)
                cnt_step += step_len

            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()


        # rgb_im.save(
        #     os.path.join(episode_data_dir, "final.png")
        # )
    
        # print(smx_vlm_rel)
        if flag == 0:
        
            success = 0
            spl = 0

            # Episode summary
            logging.info(f"\n== Episode Summary")
            logging.info(f"Scene: {scene}, Goal: {goal}")
            logging.info(f"Success : {success}, SPL : {spl}")
            # total
            if goal=="bed":
                success_bed +=success
                spl_bed +=spl
                num_bed +=1
            elif goal=="toilet":
                success_toilet +=success
                spl_toilet +=spl
                num_toilet +=1
            elif goal=="tv_monitor":
                success_tv_monitor +=success
                spl_tv_monitor +=spl
                num_tv_monitor +=1
            elif goal=="chair":
                success_chair +=success
                spl_chair +=spl
                num_chair +=1
            elif goal=="sofa":
                success_sofa +=success
                spl_sofa +=spl
                num_sofa +=1
            elif goal=="plant":
                success_plant +=success
                spl_plant +=spl
                num_plant +=1
            success_all +=success
            spl_all += spl
            logging.info(f"Success_All : {success_all/(num)}, SPL_All : {spl_all/(num)}")
            logging.info(f"== Class Summary ==")
            try:
                logging.info(f"bed: Success: {success_bed/num_bed} SPL: {spl_bed/num_bed}")
                logging.info(f"toilet: Success: {success_toilet/num_toilet} SPL: {spl_toilet/num_toilet}")
                logging.info(f"tv_monitor: Success: {success_tv_monitor/num_tv_monitor} SPL: {spl_tv_monitor/num_tv_monitor}")
                logging.info(f"chair: Success: {success_chair/num_chair} SPL: {spl_chair/num_chair}")
                logging.info(f"sofa: Success: {success_sofa/num_sofa} SPL: {spl_sofa/num_sofa}")
                logging.info(f"plant: Success: {success_plant/num_plant} SPL: {spl_plant/num_plant}")
            except Exception as e:
                logging.error(f"e: {e}")

    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
