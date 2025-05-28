import vame


# These paths have to be set manually
working_directory = "./"
project = "my-vame-project"
videos = ["./video-1.mp4"]
poses_estimations = ["./video-1.csv"]


# Step 1.1: Initialize your project
config = vame.init_new_project(
    project_name=project,
    videos=videos,
    poses_estimations=poses_estimations,
    working_directory=working_directory,
    video_type=".mp4",
)

# After the inital creation of your project you can always access the config.yaml file
# via specifying the path to your project

# Step 1.2: Align your behavior videos egocentric and create training dataset
# pose_ref_index: list of reference coordinate indices for alignment
# Example: 0: snout, 1: forehand_left, 2: forehand_right, 3: hindleft, 4: hindright, 5: tail
vame.egocentric_alignment(config, pose_ref_index=[0, 5])

# If your experiment is by design egocentrical (e.g. head-fixed experiment on treadmill etc)
# you can use the following to convert your .csv to a .npy array, ready to train vame on it
vame.csv_to_numpy(config)

# Step 1.3: create the training set for the VAME model
vame.create_trainset(
    config,
    check_parameter=False,
    pose_ref_index=[0, 5],
)

# # Step 2: Train VAME
vame.train_model(config)

# # Step 3: Evaluate model
vame.evaluate_model(config)

# # Step 4: Segment motifs/pose
vame.segment_session(config)


# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
# # The following are optional choices to create motif videos, communities/hierarchies of behavior,
# # community videos

# # OPTIONAL: Create motif videos to get insights about the fine grained poses
# vame.motif_videos(config, video_type='.mp4')

# # OPTIONAL: Create behavioural hierarchies via community detection
# vame.community(config, cut_tree=2)

# # OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
# vame.community_videos(config)

# # OPTIONAL: Down projection of latent vectors and visualization via UMAP
# vame.visualization(config, label=None) #options: label: None, "motif", "community"

# # OPTIONAL: Use the generative model (reconstruction decoder) to sample from
# # the learned data distribution, reconstruct random real samples or visualize
# # the cluster center for validation
# vame.generative_model(config, mode="centers") #options: mode: "sampling", "reconstruction", "centers", "motifs"

# # OPTIONAL: Create a video of an egocentrically aligned mouse + path through
# # the community space (similar to our gif on github) to learn more about your representation
# # and have something cool to show around ;)
# # Note: This function is currently very slow. Once the frames are saved you can create a video
# # or gif via e.g. ImageJ or other tools
# vame.gif(config, pose_ref_index=[0,5], subtract_background=True, start=None,
#          length=500, max_lag=30, label='community', file_format='.mp4', crop_size=(300,300))
