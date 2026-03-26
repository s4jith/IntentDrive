import matplotlib.pyplot as plt
from inference import predict


def plot_scene(points, neighbor_points_list=None):
    if neighbor_points_list is None:
        neighbor_points_list = []
        
    pred, probs, attn_weights = predict(points, neighbor_points_list)

    x_obs = [p[0] for p in points]
    y_obs = [p[1] for p in points]

    plt.figure(figsize=(10, 10))

    # 🚗 Car (origin reference for scale context, assuming relative to points[0])
    plt.scatter(points[0][0], points[0][1],
                c='black', s=120, marker='s', label="Car (Start Ref)")

    # 👤 Past trajectory
    plt.plot(x_obs, y_obs, 'bo-', linewidth=2, label="Target Past Path")

    # 🔥 Mark last observed point
    target_curr_x, target_curr_y = x_obs[-1], y_obs[-1]
    plt.scatter(target_curr_x, target_curr_y,
                c='blue', s=150, edgecolors='black', label="Target Current")

    # 👥 Plot Neighbors and Attention Links
    if attn_weights and attn_weights[0] is not None:
        weights = attn_weights[0].numpy()
        for i, n_pts in enumerate(neighbor_points_list):
            n_x = [p[0] for p in n_pts]
            n_y = [p[1] for p in n_pts]
            plt.plot(n_x, n_y, 'go-', linewidth=1.5, alpha=0.5)
            plt.scatter(n_x[-1], n_y[-1], c='green', s=100, edgecolors='black', alpha=0.8)
            
            # Draw attention link from target's current to neighbor's current
            w = float(weights[i])
            if w > 0.05:
                plt.plot([target_curr_x, n_x[-1]], [target_curr_y, n_y[-1]], 
                         'k--', linewidth=1 + w*5, alpha=w)
                plt.text((target_curr_x + n_x[-1])/2, (target_curr_y + n_y[-1])/2, 
                         f"{w:.2f}", fontsize=9, color='black')

    # 🎯 Predicted trajectories
    colors = ['red', 'orange', 'purple']

    for i in range(pred.shape[0]):
        x_pred = pred[i][:, 0].numpy()
        y_pred = pred[i][:, 1].numpy()

        # 🔥 fading effect (uncertainty)
        for t in range(len(x_pred)):
            alpha = 0.3 + (t / len(x_pred)) * 0.7
            plt.scatter(x_pred[t], y_pred[t],
                        color=colors[i], alpha=alpha)

        plt.plot(x_pred, y_pred,
                 color=colors[i],
                 linewidth=2 + probs[i].item()*2,
                 label=f"Pred {i+1} (p={probs[i]:.2f})")

    plt.title("Multi-Modal Trajectory Prediction with Attention", fontsize=16)
    plt.xlabel("X position (meters)")
    plt.ylabel("Y position (meters)")

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")  # 🔥 important for real-world scale
    
    fig = plt.gcf()
    
    # Save behavior for direct script runs, bypass when returning for streamlit.
    if __name__ == '__main__':
        plt.savefig("demo_plot.png", bbox_inches='tight')
        plt.show()
    
    return fig


if __name__ == "__main__":
    main_pedestrian = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3)
    ]
    
    # Adding some fake neighbors to see attention!
    neighbors = [
        [(5, 7), (5, 6), (5, 5), (5, 4)],  # Moving towards main
        [(0, 0), (0, 1), (0, 2), (0, 3)]   # Moving far away
    ]

    plot_scene(main_pedestrian, neighbors)