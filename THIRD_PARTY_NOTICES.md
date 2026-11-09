# Third-party notices

The MIT licence in [`LICENSE`](LICENSE) covers the original code and assets in
this repository. The components below are included under their own licences,
which take precedence for those files.

## Code

| Component | Path | Licence | Source |
|---|---|---|---|
| ROS-TCP-Endpoint | `docker/ros_server/ROS/src/ros_tcp_endpoint` | Apache 2.0 | [Unity-Technologies/ROS-TCP-Endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint) |
| ROS-TCP-Connector 0.1.2-preview | `unity/Packages/com.unity.robotics.ros-tcp-connector` | Apache 2.0 | [Unity-Technologies/ROS-TCP-Connector](https://github.com/Unity-Technologies/ROS-TCP-Connector) |
| `roboracer` package (message and node scaffolding, originally `niryo_moveit`) | `docker/ros_server/ROS/src/roboracer` | Apache 2.0 | [Unity-Technologies/Unity-Robotics-Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) pick-and-place tutorial |
| rospy_message_converter | `docker/ros_server/ROS/src/rospy_message_converter` | BSD | [uos/rospy_message_converter](https://github.com/uos/rospy_message_converter) |

The C# message classes under `unity/Assets/RosMessages` are generated from the
message definitions above.

### Git submodules

These are separate repositories pinned as submodules (forks of the upstream
projects), not files of this repository. Their own licence files apply.

| Component | Path | Licence | Source |
|---|---|---|---|
| microduck_rl (Microduck RL training environments) | `third_party/microduck_rl` | Apache 2.0, except the STL meshes (below) | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) |
| microduck (Microduck robot runtime) | `third_party/microduck` | Apache 2.0 | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) |

The Microduck STL meshes (`third_party/microduck_rl/src/mjlab_microduck/robot/*/assets/*.stl`)
are © Pollen Robotics under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
non-commercial use only, with attribution, and adaptations under the same
licence. See that repository's `LICENSE-MODELS.md`. They are not included in
this project's release archives.

## Unity Asset Store content

These files come from Unity Asset Store packages and are governed by the
[Unity Asset Store EULA](https://unity.com/legal/as-terms), not by the MIT
licence. Only the files the course scenes use are included. They may not be
extracted or reused outside this project; get your own copy from the Asset
Store to build on them.

| Package | Path |
|---|---|
| Race Track Construction Kit | `unity/Assets/Race_Track_Construction_Kit` |
| Rocks | `unity/Assets/Rocks` |

## 3D models

| Model | Path | Licence |
|---|---|---|
| Fruit fly (male) | `unity/Assets/Resources/FruitFly/*.fbx`, `unity/Assets/FruitFly/Textures` | [TurboSquid Royalty Free License](https://blog.turbosquid.com/royalty-free-license/); not covered by the MIT licence |
