import streamlit as st
import matplotlib.pyplot as plt
from visualization import plot_scene

st.set_page_config(page_title="Hackathon Demo: Trajectory Prediction", layout="wide")

st.title("🚗 Social-Aware Multi-Modal Trajectory Prediction")
st.markdown("""
Welcome to the interactive demo! This model looks at 2 seconds of past pedestrian motion, and predicts 3 distinct potential future paths for the next 3 seconds. 

It uses an **Attention Mechanism** to "look" at neighboring pedestrians to adjust its predictions and avoid collisions.
""")

st.sidebar.header("Scenario Configuration")

scenario = st.sidebar.selectbox("Select a Demo Scenario", [
    "Clean Straight Path (No interaction)",
    "Approaching Neighbor (Social impact)",
    "Crowded Scene",
    "Custom Coordinate Input"
])

# Base configuration for main pedestrian
main_pedestrian = [(2, 3), (3, 3), (4, 3), (5, 3)]
neighbors = []

if scenario == "Clean Straight Path (No interaction)":
    st.markdown("### Scenario: Clear Path")
    st.info("The pedestrian is walking straight with no neighboring individuals. The model predicts they will continue straightforward with slight deviations.")

elif scenario == "Approaching Neighbor (Social impact)":
    st.markdown("### Scenario: Approaching Neighbor")
    st.info("Notice how the pedestrian avoids the neighbor walking head-on towards them!")
    # Neighbor moving left from top right, towards the target's future path
    neighbors = [[(8, 5), (7, 4), (6, 3), (5, 2)]]

elif scenario == "Crowded Scene":
    st.markdown("### Scenario: Crowded Interaction")
    st.info("Multiple people navigating the space. Look at the variable attention thicknesses!")
    neighbors = [
        [(8, 5), (7, 4), (6, 3), (5, 2)],   # Close interaction
        [(1, 1), (1, 2), (1, 3), (1, 4)],   # Behind, moving up
        [(3, 7), (4, 7), (5, 7), (6, 7)]    # Moving parallel but far away
    ]

elif scenario == "Custom Coordinate Input":
    st.markdown("### Scenario: Custom Coordinate Input")
    st.info("Provide exactly 4 historical (x,y) points for the pedestrian. The last point is the current position (t=0). Format: x,y separated by semicolons.")
    
    st.markdown("**Example Custom Setup:**")
    raw_main = st.text_input("Main Pedestrian Points (t=-3, t=-2, t=-1, t=0)", "2,3; 3,3; 4,3; 5,3")
    raw_neighbors = st.text_area("Neighbor Points (One neighbor per line)", "8,5; 7,4; 6,3; 5,2\n3,7; 4,7; 5,7; 6,7")
    
    try:
        parsed_main = []
        for pt in raw_main.strip().split(';'):
            if pt.strip():
                x, y = map(float, pt.strip().split(','))
                parsed_main.append((x, y))
        
        if len(parsed_main) != 4:
            st.error(f"Your main pedestrian trajectory must have exactly 4 points. You provided {len(parsed_main)}.")
            st.stop()
            
        main_pedestrian = parsed_main
        neighbors = []
        if raw_neighbors.strip():
            for line in raw_neighbors.strip().split('\n'):
                if line.strip():
                    n_pts = []
                    for pt in line.strip().split(';'):
                        if pt.strip():
                            x, y = map(float, pt.strip().split(','))
                            n_pts.append((x, y))
                    if len(n_pts) != 4:
                        st.error(f"Each neighbor must have exactly 4 points. Faulty line has {len(n_pts)}: {line}")
                        st.stop()
                    neighbors.append(n_pts)
    except Exception as e:
        st.error(f"Error parsing coordinates: {e}. Please use format 'x,y; x,y; x,y; x,y'")
        st.stop()

# Tweak target speed optionally in sidebar
if scenario != "Custom Coordinate Input":
    speed_multiplier = st.sidebar.slider("Target Speed Multiplier", min_value=0.5, max_value=2.0, value=1.0)
    adjusted_main = [(2, 3), (2 + 1*speed_multiplier, 3), (2 + 2*speed_multiplier, 3), (2 + 3*speed_multiplier, 3)]
else:
    adjusted_main = main_pedestrian


with st.spinner("Running Inference and Generating Scene..."):
    fig = plot_scene(adjusted_main, neighbors)
    st.pyplot(fig)

st.markdown("---")
st.markdown("### Hackathon Team: [Your Name/Team]")
st.markdown("Built using **PyTorch**, **nuScenes**, and **Attention-LSTM**.")
