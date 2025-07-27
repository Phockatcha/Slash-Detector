# Slash-Detector
Find the coordinates of handwritten marks on a multi-column register to assign each individual to their corresponding class. Example based on the economic condition of citizens in a 19th-century Italian census.

This workstream aims to identify visual markers and their coordinates across the full page of a historical register (see example below). Depending on how it is adapted, this pipeline can serve multiple purposes, but its core function is to detect handwritten marks that indicate categorical variables.

<img width="940" height="696" alt="image" src="https://github.com/user-attachments/assets/febd6eae-51e2-4b6c-ab4f-5f6583ea8c1e" />

In this specific case, the categorization is column-based and refers to the second table on the right-hand side of the image ("Segno Economico"). Each mark in this table denotes a category: "Ricco", "Benestante", "Comodo", "Povero", or "Indigente".

Step 1. Cropping the Full-Size Image (Crop_FullSize.py)
Goal: to restrict the detection area to the economic table only, speeding up computation and avoiding false positives from other regions. This is done by defining normalized crop coordinates based on the near-constant layout of the scanned pages.
Example: 

<img width="940" height="539" alt="image" src="https://github.com/user-attachments/assets/9ff85361-f20b-4558-af14-886a96a1e9b7" />

Step 2. Grid Line Extraction (Grid_Masks.py)
Goal: to detect the vertical and horizontal grid lines of the economic table using morphological operations, and exclude them from subsequent analysis. This ensures that table lines are not misinterpreted as handwritten marks (meaning, economic slashes).
Workstream:
- Convert the cropped image to grayscale.
- Apply inverted adaptive thresholding.
- Use morphological kernels (vertical lines: 1×35, horizontal lines: 35×1)
- Save the output: a binary mask highlighting grid lines.
Example:

<img width="939" height="734" alt="image" src="https://github.com/user-attachments/assets/8b2c2797-2beb-43d2-b67e-2b53343b3f55" />

Step 3. Grid Line Interpolation (Gridline_Interp.py)
Goal: to cluster vertical segments detected via Hough transform and compute a single average line per group, to define a precise estimate of table boundaries. Hough Lines often return multiple close lines due to small curvature or noise, but for precision, a single pixel-wide line per grid line is needed.
Workstream:
- Apply cv2.HoughLinesP on each grid mask.
- Filter lines with angles close to 90° (±1°) to exclude slanted segments.
- Cluster lines based on their average X coordinate (with a tolerance X_CLUSTER_PCT).
- Compute average intersection points (top and bottom).
- Draw and save the final lines for use in later steps.
Example: 

<img width="797" height="600" alt="image" src="https://github.com/user-attachments/assets/0f4b7e76-87aa-4eb6-ab79-ff660db1dfd6" />

Step 4. Slash Detector (Economic_Marks.py)
Goal: to identify blobs representing economic marks, apply a series of filters, and save validated results to CSV. (Note: for visualization, refer to Step 5. The viewer is implemented as a separate script for easier debugging and review).
Workstream:
For each cropped image:
(1) Load the image and convert it to grayscale.
(2) Load the corresponding interpolated grid lines mask (from Step 3).
(3) Extract left/right X coordinates for each column, using the mask. These define valid column boundaries.
(4) Dilate the grid mask horizontally by GRID_BUFFER_PCT to eliminate blobs too close to the table lines.
(5) Create a binary image excluding original grid pixels.
Then, for each detected blob define criteria for exclusion:
- Size filter: discard blobs too small in width/height.
- Height filter: discard blobs that are too tall (likely stains).
- grid proximity filter: discard blobs fully inside the buffered grid zone.
- bottom noise filter: discard blobs too close to the bottom edge (where dark noise often accumulates). Blobs that pass all checks are added to primary_pass.
Additional logic:
- Discard blobs whose center falls outside the horizontal bounds of any valid column (stored as discarded_A).
- Then, check if any blob inside the column lies near the border (within WORD_BUFFER_PCT) and is close to a blob in discarded_A — likely a protruding character. These are discarded too.
- Detect blobs that are too vertically close (gap below CLOSE_DIST_PCT) — possibly indicating an error. These files are flagged and stored in images_flagged.
Output are produced in two versions, to help reviewing potential errors.
- A CSV with validated slash bounding boxes and coordinates.
- A CSV listing images with suspected errors (too-close blobs).

Step 5. Slash Visualization Tool  (VisualizerX.py)
Goal: to generate annotated images with coloured bounding boxes, indicating the reason each blob was kept or discarded (or alternatively, showing only validated slashes). Useful for manual inspection and debugging.
Tool – Version 1 (only validated – Visualizer1.py). Example:

<img width="870" height="677" alt="image" src="https://github.com/user-attachments/assets/9a391ca4-8cda-4183-be2f-227961ae17a2" />
 
Tool – Version 2 (debug and review – Visualized2.py). Example:
 
<img width="805" height="666" alt="image" src="https://github.com/user-attachments/assets/e02b4b9f-1375-4fcf-b2b8-a5486c29e256" />
