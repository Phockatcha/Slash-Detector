# Requirements for Pip and folder hierarchy

pip install numpy
pip install opencv-python

# Folder Hierarchy

#├── data/
#   ├── raw/                         # Full Size Images
#   ├── crop/                        # Crop (Step 1)
#  │   ├── grid_masks/              # Grid mask extractor (Step 2)
#   │   ├── grid_lines_hough_avg/    # Hough interpolation (Step 3)
#   │   ├── vis_1/                   # Visualization only validated (Step 5.1)
#   │   └── vis_2/                   # Visualization debug (Step 5.2)
#   └── output/
#       ├── slash_boxes_detector.csv      # Final bounding box, kept and discarded (Step 4)
#       └── images_close_blobs.csv        # Suspected errors (Step 4)