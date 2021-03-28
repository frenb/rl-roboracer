import asyncio
import math

from api import RobotApi
from utility import camera_to_niryo_coords
from utility import frame_to_tensor
from utility import tf_load_model
from utility import tf_detect
from utility import tf_result_to_list
from utility import annotate_camera


# Banana class ID from SSD-MobileNet
BANANA_CLASS = 52

# Quaternion peripindicular to the niryo table.
TABLE_PERPINDICULAR = {'x' : -0.5, 'y' : -0.5, 'z' : 0.5, 'w' : -0.5}

async def main():
    api = RobotApi()
    await api.Initialize()
    print("API Initialized")
    
    detector = await tf_load_model("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    print("Detection Model loaded")

    # Save initial pose for restoring robot.
    scene_data = await api.GetSceneData()
    initial_pose = scene_data['effector_pose']

    while True:
        # Get a new frame from the overhead camera and run inference
        # to detect bananas.
        img = await api.GetOverheadCameraFrame()
        img = frame_to_tensor(img)
        
        results = await tf_detect(detector, img)
        bananas = list(
            filter(
                lambda r: r['class_id'] == BANANA_CLASS and r['score'] >= 0.1,
                tf_result_to_list(results)))
        
        if len(bananas) == 0:
            annotate_camera(boxes=[])
            print("No banana. We sleep.")
            continue
        
        # Display a box around the banana in the IDE.
        print("A banana! This can not stand.")
        bbox = bananas[0]['bbox']
        annotate_camera(boxes=[bbox])

        # Calculate banana bounds in Niryo coordindates.
        # Pick the closest coordinate to begin the push.
        x_min = bbox[1]
        y_min = bbox[0]
        x_max = bbox[3]
        y_max = bbox[2]
        corners = [
            camera_to_niryo_coords((x_min, 1 - y_min)),
            camera_to_niryo_coords((x_min, 1 - y_max)),
            camera_to_niryo_coords((x_max, 1 - y_min)),
            camera_to_niryo_coords((x_max, 1 - y_max)),
        ]
        corners.sort(key=lambda point: point[0]**2 + point[1]**2)
        
        # Execute start trajectory.
        position = {
            'x': corners[0][0],
            'y': corners[0][1],
            'z': 0.7
        }
        goal = {'position': position, 'orientation': TABLE_PERPINDICULAR}
        plan = await api.GetPlan(goal)
        await api.DoTrajectory(plan['trajectory'])

        # Execute push trajectory.
        r_max = 0.36 ** 2
        r_corner = corners[3][0] ** 2. +corners[3][1] ** 2
        m = math.sqrt(r_max / r_corner)
        position = {
            'x': m * corners[3][0],
            'y': m * corners[3][1],
            'z': 0.7
        }
        goal = {'position': position, 'orientation': TABLE_PERPINDICULAR}
        plan = await api.GetPlan(goal)
        await api.DoTrajectory(plan['trajectory'])

        # Executre return trajectory.
        plan = await api.GetPlan(initial_pose)
        await api.DoTrajectory(plan['trajectory'])

        

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())