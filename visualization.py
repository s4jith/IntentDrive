import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from inference import predict
from map_renderer import render_map_patch

def plot_scene(
    points,
    neighbor_points_list=None,
    neighbor_types=None,
    is_live_camera=False,
    sensor_fusion=None,
    presentation_mode=False,
    max_vru_display=6,
):
    if neighbor_points_list is None: sibling_pts = []
    else: sibling_pts = neighbor_points_list
        
    if neighbor_types is None: n_types = ['Car'] * len(sibling_pts)
    else: n_types = neighbor_types

    # Set up dark "Extreme 3D Mode" environment if it's Live Camera
    plt.style.use('dark_background') if is_live_camera else plt.style.use('default')
    fig = plt.figure(figsize=(14, 12))
    ax = plt.gca()
    
    # ---------------- EGO VEHICLE & CAMERA PERSPECTIVE ----------------
    if is_live_camera:
        # In live camera mode, we anchor the BEV map to the Ego car!
        ego_x, ego_y = 0.0, -2.0
        ax.set_facecolor('#0b0e14') 
        
        # Add Compass Directions
        ax.text(0, 48, "N (Forward)", color="white", fontsize=14, weight="bold", ha="center")
        ax.text(0, -8, "S (Rear)", color="white", fontsize=14, weight="bold", ha="center", alpha=0.5)
        ax.text(32, ego_y, "E (Right)", color="white", fontsize=14, weight="bold", ha="left", alpha=0.5)
        ax.text(-32, ego_y, "W (Left)", color="white", fontsize=14, weight="bold", ha="right", alpha=0.5)
        
        plt.grid(True, linestyle='dotted', color='#1a2436', alpha=0.9, zorder=0)
        
        theta = np.linspace(np.pi/3, 2 * np.pi/3, 50)
        fov_range = 60
        ax.fill_between(
            [ego_x] + list(ego_x + fov_range * np.cos(theta)) + [ego_x],
            [ego_y] + list(ego_y + fov_range * np.sin(theta)) + [ego_y],
            color='#00ffff', alpha=0.1, zorder=1, label='360 Camera / LiDAR FOV'
        )
        car_rect = patches.Rectangle((ego_x - 1.2, ego_y - 2.5), 2.4, 5.0, linewidth=2, edgecolor='#00ffff', facecolor='#001a1a', zorder=7, label="Autonomous Ego Vehicle")
        ax.add_patch(car_rect)
        
        ax.set_xlim(-35, 35)
        ax.set_ylim(-10, 50)
        map_center_x, map_center_y = 0, 20
        ego_ref = np.array([ego_x, ego_y], dtype=np.float32)
    else:
        map_center_x, map_center_y = points[-1][0], points[-1][1]
        ego_x, ego_y = map_center_x - 12, map_center_y - 6
        theta = np.linspace(-np.pi/6, np.pi/6, 50)
        ax.fill_between(
            [ego_x] + list(ego_x + 50 * np.cos(theta)) + [ego_x],
            [ego_y] + list(ego_y + 50 * np.sin(theta)) + [ego_y],
            color='cyan', alpha=0.15, zorder=2
        )
        car_rect = patches.Rectangle((ego_x - 2.4, ego_y - 1.0), 4.8, 2.0, linewidth=2, edgecolor='black', facecolor='cyan', zorder=7)
        ax.add_patch(car_rect)
        ax.set_xlim(map_center_x - 15, map_center_x + 35)
        ax.set_ylim(map_center_y - 20, map_center_y + 20)
        plt.grid(True, linestyle='solid', color='lightgray', alpha=0.5, zorder=1)
        ego_ref = np.array([map_center_x, map_center_y], dtype=np.float32)

    if not is_live_camera:
        render_map_patch(map_center_x, map_center_y, radius=120.0, ax=ax)

    # ---------------- Phase 1 Sensor Fusion Overlay ----------------
    if is_live_camera and sensor_fusion is not None:
        lidar_xy = sensor_fusion.get('lidar_xy', None)
        radar_xy = sensor_fusion.get('radar_xy', None)
        radar_vel = sensor_fusion.get('radar_vel', None)

        if lidar_xy is not None and len(lidar_xy) > 0:
            # Remove very-near ego returns to avoid halo clutter around the car.
            r = np.hypot(lidar_xy[:, 0] - ego_ref[0], lidar_xy[:, 1] - ego_ref[1])
            lidar_vis = lidar_xy[r > 6.0]

            if presentation_mode:
                step = 18 if len(lidar_vis) > 12000 else 10
                lidar_plot = lidar_vis[::step] if len(lidar_vis) > 0 else lidar_vis
                lidar_size = 3
                lidar_alpha = 0.10
            else:
                lidar_plot = lidar_vis[::4] if len(lidar_vis) > 4000 else lidar_vis
                lidar_size = 5
                lidar_alpha = 0.18

            ax.scatter(
                lidar_plot[:, 0],
                lidar_plot[:, 1],
                s=lidar_size,
                c='#22d3ee',
                alpha=lidar_alpha,
                linewidths=0,
                label='LiDAR occupancy',
                zorder=2,
            )

        if radar_xy is not None and len(radar_xy) > 0:
            if presentation_mode and len(radar_xy) > 180:
                radar_plot = radar_xy[::2]
            else:
                radar_plot = radar_xy

            ax.scatter(
                radar_plot[:, 0],
                radar_plot[:, 1],
                s=18 if presentation_mode else 24,
                c='#facc15',
                alpha=0.78 if presentation_mode else 0.85,
                edgecolors='black',
                linewidths=0.5,
                label='Radar returns (multi-ch)',
                zorder=6,
            )

            if radar_vel is not None and len(radar_vel) == len(radar_xy):
                speeds = np.hypot(radar_vel[:, 0], radar_vel[:, 1])
                if presentation_mode:
                    idx = np.where(speeds > 0.6)[0]
                    if len(idx) > 18:
                        idx = idx[np.argsort(speeds[idx])[-18:]]
                else:
                    step = max(1, len(radar_xy) // 40)
                    idx = np.arange(0, len(radar_xy), step)

                for i in idx:
                    x0, y0 = radar_xy[i, 0], radar_xy[i, 1]
                    vx, vy = radar_vel[i, 0], radar_vel[i, 1]
                    ax.arrow(
                        x0,
                        y0,
                        vx * (0.45 if presentation_mode else 0.6),
                        vy * (0.45 if presentation_mode else 0.6),
                        head_width=0.45 if presentation_mode else 0.6,
                        head_length=0.6 if presentation_mode else 0.8,
                        fc='#fde68a',
                        ec='#facc15',
                        alpha=0.65 if presentation_mode else 0.75,
                        zorder=6,
                        length_includes_head=True,
                    )

    # ---------------- MULTI-AGENT PREDICTIONS ----------------
    color_map = {'Car': '#ffff00', 'Truck': '#ffaa00', 'Bus': '#ff8800', 'Person': '#ff00ff', 'Bike': '#ff5500'}

    def build_agent_fusion_features(agent_points):
        if sensor_fusion is None:
            return None

        lidar_xy = sensor_fusion.get('lidar_xy', None)
        radar_xy = sensor_fusion.get('radar_xy', None)

        if lidar_xy is None and radar_xy is None:
            return None

        feats = []
        for px, py in agent_points:
            if lidar_xy is not None and len(lidar_xy) > 0:
                dl = np.hypot(lidar_xy[:, 0] - px, lidar_xy[:, 1] - py)
                lidar_cnt = int((dl < 2.0).sum())
            else:
                lidar_cnt = 0

            if radar_xy is not None and len(radar_xy) > 0:
                dr = np.hypot(radar_xy[:, 0] - px, radar_xy[:, 1] - py)
                radar_cnt = int((dr < 2.5).sum())
            else:
                radar_cnt = 0

            lidar_norm = min(80.0, float(lidar_cnt)) / 80.0
            radar_norm = min(30.0, float(radar_cnt)) / 30.0
            sensor_strength = min(1.0, (float(lidar_cnt) + 2.0 * float(radar_cnt)) / 100.0)
            feats.append([lidar_norm, radar_norm, sensor_strength])

        return feats

    def classify_mode_direction(hist_x, hist_y, pred_x, pred_y):
        if len(hist_x) < 2:
            return 'Straight'

        # Current motion heading from the last observed segment.
        hx = hist_x[-1] - hist_x[-2]
        hy = hist_y[-1] - hist_y[-2]
        if np.hypot(hx, hy) < 1e-6:
            hx, hy = 0.0, 1.0

        # Predicted heading from current point to mode endpoint.
        px = pred_x[-1] - hist_x[-1]
        py = pred_y[-1] - hist_y[-1]
        if np.hypot(px, py) < 1e-6:
            return 'Straight'

        angle_deg = np.degrees(np.arctan2(hx * py - hy * px, hx * px + hy * py))

        if abs(angle_deg) <= 30:
            return 'Straight'
        if 30 < angle_deg < 140:
            return 'Left'
        if -140 < angle_deg < -30:
            return 'Right'
        return 'Backward'

    all_agents_to_predict = [(points, 'Person (Primary)')]
    for i, n_pts in enumerate(sibling_pts):
        # We now run predictions for ANY vulnerable user (Person or Bicycle)
        if is_live_camera and n_types[i] in ['Person', 'Bicycle']:
            all_agents_to_predict.append((n_pts, f"{n_types[i]} {i}"))

    # Keep the live demo readable by limiting displayed VRUs in presentation mode.
    if is_live_camera and presentation_mode and len(all_agents_to_predict) > max_vru_display:
        primary = all_agents_to_predict[0]
        others = all_agents_to_predict[1:]

        def _dist_to_ego(agent_entry):
            pts = agent_entry[0]
            if len(pts) == 0:
                return 1e9
            px, py = pts[-1][0], pts[-1][1]
            return float(np.hypot(px - ego_ref[0], py - ego_ref[1]))

        others = sorted(others, key=_dist_to_ego)
        all_agents_to_predict = [primary] + others[: max(0, max_vru_display - 1)]

    vru_mode_summaries = []
    vru_counter = 1

    # Predict and plot the future for all identified vulnerable users
    for agent_pts, label in all_agents_to_predict:
        fusion_feats = build_agent_fusion_features(agent_pts)
        pred, probs, attn_weights = predict(agent_pts, sibling_pts, fusion_feats=fusion_feats)
        tx, ty = [p[0] for p in agent_pts], [p[1] for p in agent_pts]
        is_primary = 'Primary' in label
        mode_direction_scores = {}
        
        # Plot their history (tail)
        plt.plot(tx, ty, color='white' if is_primary else '#ff00ff', linestyle='solid' if is_live_camera else 'dashed', linewidth=3, zorder=5)
        if is_live_camera:
            point_label = 'Primary VRU (t=0)' if is_primary else 'Target VRU (t=0)'
        else:
            point_label = f"{label} (t=0)"
        plt.scatter(tx[-1], ty[-1], c='white' if is_primary else '#ff00ff', s=250 if is_primary else 150, edgecolors='black', linewidths=2, label=point_label, zorder=8)
        
        # --- NEW: Add an extremely obvious Vector Arrow showing their Current Walking Direction ---
        if len(tx) >= 2:
            dx_dir = tx[-1] - tx[-2]
            dy_dir = ty[-1] - ty[-2]
            dir_mag = np.hypot(dx_dir, dy_dir)
            if dir_mag > 0.01:
                # The arrow dynamically scales to their movement speed and points exactly where they are headed!
                arr_dx, arr_dy = (dx_dir/dir_mag)*3, (dy_dir/dir_mag)*3
                ax.arrow(tx[-1], ty[-1], arr_dx, arr_dy, head_width=1.5, head_length=2.0, fc='#00ffff', ec='white', zorder=12, width=0.4, alpha=0.9)
                
        # Plot their Future prediction paths
        colors = ['#0088ff', '#ff8800', '#ff0044']
        mode_curves = []
        
        for mode_i in range(pred.shape[0]):
            x_pred_raw = pred[mode_i][:, 0].numpy()
            y_pred_raw = pred[mode_i][:, 1].numpy()

            dx = x_pred_raw - x_pred_raw[0]
            dy = y_pred_raw - y_pred_raw[0]
            
            x_pred = tx[-1] + dx * (2.0 if is_live_camera else 4.0)
            y_pred = ty[-1] + dy * (2.0 if is_live_camera else 4.0)
            mode_curves.append((mode_i, x_pred, y_pred))

            mode_direction = classify_mode_direction(tx, ty, x_pred, y_pred)
            mode_prob = float(probs[mode_i].item())
            mode_direction_scores[mode_direction] = mode_direction_scores.get(mode_direction, 0.0) + mode_prob

        if presentation_mode and is_live_camera:
            draw_modes = [int(np.argmax(probs.numpy()))]
        else:
            draw_modes = [m[0] for m in mode_curves]

        for mode_i, x_pred, y_pred in mode_curves:
            if mode_i not in draw_modes:
                continue
            plt.plot(
                x_pred,
                y_pred,
                color=colors[mode_i],
                linewidth=3.0 if presentation_mode else 2.5 + (0 if mode_i > 0 else 1),
                alpha=0.9 if presentation_mode else (0.8 if mode_i == 0 else 0.4),
                zorder=5,
            )
            for t in range(0, len(x_pred), 3 if presentation_mode else 2):
                plt.scatter(
                    x_pred[t],
                    y_pred[t],
                    color=colors[mode_i],
                    alpha=max(0.35, 1.0 - (t / 12)),
                    s=28 if presentation_mode else 40,
                    zorder=6,
                )

        # Per-agent Top-3 direction probabilities for live demo readability.
        sorted_modes = sorted(mode_direction_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_modes = sorted_modes[:3]
        vru_id = f"VRU-{vru_counter}" + ("*" if is_primary else "")
        vru_mode_summaries.append((vru_id, top_modes))

        if is_live_camera and (not presentation_mode) and len(top_modes) > 0:
            primary_dir, primary_prob = top_modes[0]
            ax.text(
                tx[-1] + 0.8,
                ty[-1] + 1.2,
                f"{vru_id}: {primary_dir} {primary_prob*100:.0f}%",
                fontsize=8,
                color='white',
                bbox=dict(facecolor='#111827', edgecolor='#60a5fa', alpha=0.8, boxstyle='round,pad=0.2'),
                zorder=13
            )
        vru_counter += 1

    # ---------------- PLOT NEIGHBORS (Vehicles/Trucks) ----------------
    for i, n_pts in enumerate(sibling_pts):
        if is_live_camera and n_types[i] in ['Person', 'Bicycle']:
            continue # Already predicted above
            
        n_type = n_types[i]
        n_color = color_map.get(n_type, 'yellow')
        n_x, n_y = [p[0] for p in n_pts], [p[1] for p in n_pts]
        
        marker_size = 400 if n_type in ['Truck', 'Bus'] else 200
        marker_shape = 's' if n_type in ['Truck', 'Bus'] else 'o'
        
        plt.plot(n_x, n_y, color=n_color, linestyle=':', linewidth=2, zorder=4)
        plt.scatter(n_x[-1], n_y[-1], c=n_color, marker=marker_shape, s=marker_size, edgecolors='white' if is_live_camera else 'black', linewidth=1.5, label=f'Moving ({n_type})', zorder=7)

    # UI Embellishments
    plt.title("Ego-Centric BEV Matrix: Multi-Agent Parallel Forecasting", color="white" if is_live_camera else "black", fontsize=20, weight='bold', pad=15)
    plt.xlabel("X Lateral Offset (meters)", color="white" if is_live_camera else "black", weight='bold', fontsize=13)
    plt.ylabel("Y Depth Offset (meters)", color="white" if is_live_camera else "black", weight='bold', fontsize=13)

    if is_live_camera:
        ax.tick_params(axis='both', colors='white', labelsize=11)
        for spine in ax.spines.values():
            spine.set_color('#94a3b8')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    if is_live_camera:
        leg = ax.legend(
            unique_handles,
            unique_labels,
            loc='upper right',
            fancybox=True,
            framealpha=0.95,
            facecolor='#111827',
            edgecolor='#94a3b8',
            fontsize=10,
            title='Legend'
        )
        plt.setp(leg.get_texts(), color='white')
        plt.setp(leg.get_title(), color='white', weight='bold')

        if len(vru_mode_summaries) > 0:
            summary_lines = ["Top-3 Direction Probabilities"]
            summary_lines.append("VRU-* = primary target")
            for vru_id, top_modes in vru_mode_summaries[:max_vru_display]:
                mode_text = " | ".join([f"{name}:{prob*100:.0f}%" for name, prob in top_modes])
                summary_lines.append(f"{vru_id} -> {mode_text}")

            fig.subplots_adjust(right=0.80)
            ax.text(
                1.02,
                0.62,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                color='white',
                bbox=dict(facecolor='#0f172a', edgecolor='#60a5fa', alpha=0.95, boxstyle='round,pad=0.4')
            )
    else:
        leg = ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), fancybox=True, framealpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    return fig

if __name__ == "__main__":
    main_pedestrian = [(0, 0), (10, 0), (20, 0), (30, 0)]
    plot_scene(main_pedestrian, is_live_camera=True)
