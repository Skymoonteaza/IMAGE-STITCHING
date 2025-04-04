import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# Settings for final panorama size (20 inches wide, 1:2.5 ratio)
final_width = 5080          # 20 inches wide (in pixels at a chosen DPI)
final_height = int(final_width / 2.5)  # Height for a 1:2.5 aspect ratio (2032 pixels)

# List of image file names (all 5 images)
image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
images = [cv2.imread(img) for img in image_files]

# --------------------------------------------------------
# Choose which stitching method to use:
# Set USE_AUTOMATIC = True to use OpenCV's built-in Stitcher.
# Set USE_AUTOMATIC = False to use the manual ORB & BFMatcher method.
USE_AUTOMATIC = True
# --------------------------------------------------------

if USE_AUTOMATIC:
    # --- Method 1: Automatic Stitching using OpenCV Stitcher ---
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        print("Automatic stitching completed successfully.")
    else:
        print("Automatic stitching failed. Falling back to manual stitching.")
        USE_AUTOMATIC = False  # Switch to manual method if automatic fails

if not USE_AUTOMATIC:
    # --- Method 2: Manual Stitching using ORB feature detection and BFMatcher ---
    def resize_image(image, width=1000):
        """ Resize image to a given width while maintaining aspect ratio. """
        height, orig_width, _ = image.shape
        aspect_ratio = height / orig_width
        new_height = int(width * aspect_ratio)
        return cv2.resize(image, (width, new_height))
    
    # Resize images to ensure they are similar in size before stitching
    resized_images = [resize_image(img, 1000) for img in images]
    
    # Detect features using ORB
    orb = cv2.ORB_create(nfeatures=3000)
    keypoints_descriptors = [orb.detectAndCompute(img, None) for img in resized_images]
    
    # Create BFMatcher for matching features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def get_homography(kp1, des1, kp2, des2):
        """ Computes homography using ORB feature matching. """
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
    
        # Take the best 15% of matches (at least 10 matches)
        num_good_matches = max(10, int(len(matches) * 0.15))
        good_matches = matches[:num_good_matches]
    
        if len(good_matches) < 4:
            return None  # Not enough matches to compute homography
    
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    # Start with the first image as the base for the panorama
    panorama = resized_images[0]
    
    for i in range(1, len(resized_images)):
        kp1, des1 = keypoints_descriptors[i - 1]
        kp2, des2 = keypoints_descriptors[i]
    
        H = get_homography(kp1, des1, kp2, des2)
        if H is None:
            print(f"Skipping image {i+1}, not enough matches.")
            continue
    
        # Warp the next image onto the current panorama canvas
        height, width, _ = panorama.shape
        warped_image = cv2.warpPerspective(resized_images[i], H, (width * 2, height))
    
        # Create a larger canvas to fit both images
        canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
        canvas[:height, :width] = panorama
    
        # Blend images by replacing canvas pixels with non-black pixels from warped_image
        non_black_pixels = (warped_image > 0).any(axis=2)
        canvas[non_black_pixels] = warped_image[non_black_pixels]
    
        panorama = canvas  # Update panorama with the new canvas
    
    # Crop black edges from the stitched panorama
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray_panorama > 0))
    x, y, w, h = cv2.boundingRect(coords)
    panorama = panorama[y:y+h, x:x+w]

# --------------------------------------------------------
# Resize final panorama to the desired size (5080 x 2032 pixels)
panorama_resized = cv2.resize(panorama, (final_width, final_height))

# Display the final panorama
plt.figure(figsize=(20, 8))
plt.imshow(cv2.cvtColor(panorama_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the final panorama image
cv2.imwrite("panorama_20inch.jpg", panorama_resized)







