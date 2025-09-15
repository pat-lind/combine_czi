import argparse
from pylibCZIrw import czi
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import numpy as np
from czitools.metadata_tools.czi_metadata import CziChannelInfo
from scipy.signal import find_peaks
import time
import os
import re


def get_masks(img1, img2):
    #calculate threshold of which to draw our section mask.
    #loosely based off observations of histograms.

    img1 = process_mask(img1, display=False)

    img2 = 255-process_mask(img2, display=False) #nissl, which has different image values so we minus 255


    return img1, img2

def calculate_thresh(img):
    histogram, bins = np.histogram(img.ravel(), bins=256, range=(0,256))

    return histo_algo(histogram)


def histo_algo(histogram):
    peaks, _ = find_peaks(histogram, height=300)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    print("sorted peaks",sorted_peaks)
    top_two_peaks = sorted_peaks[:2]
    if len(top_two_peaks) == 1: return top_two_peaks[0]-1
    elif abs(top_two_peaks[0] - top_two_peaks[1]) > 15: # if their distance is pretty big, then just return the first peak. In our grayscale images there are two big populations of white and black pixels, resembling our two greatest peaks. we want whatever grayscale value of these to be our threshold.
        return top_two_peaks[0]

    else:
        return (top_two_peaks[0]+top_two_peaks[1])/2

def process_mask(image, display=False):
    img = image.copy()
    thresh  = calculate_thresh(image)
    print(thresh)
    mask = img >= thresh
    img[mask] = 255

    mask = img < thresh
    img[mask] = 0


    if display:
        fig, ax = plt.subplots(1,1, figsize = (12,32))
        ax.imshow(img, cmap='grey')
        ax.set_title("image_mask")
        plt.show()

    return img

def find_masks(fixed, moving):

    fixed_mask, moving_mask  = get_masks(fixed, moving)

    return fixed_mask, moving_mask

def downscale_image(image, downscale_factor):
    #downscales an image while updating its spacing

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(sz / downscale_factor) for sz in original_size]
    new_spacing = [sp * downscale_factor for sp in original_spacing]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)

    #print(f"    Downscaling from {original_size} to {new_size}")
    return resampler.Execute(image)


def register_images(fixedimg, movingimg):
    fixed = sitk.GetImageFromArray(fixedimg, sitk.sitkFloat64)
    moving = sitk.GetImageFromArray(movingimg, sitk.sitkFloat64)

    fixed.SetSpacing((.908, .908))
    moving.SetSpacing((.908, .908))

    downscale_factor = 32
    downscaled_fixed = downscale_image(fixed, downscale_factor)
    downscaled_moving = downscale_image(moving, downscale_factor)
    downscaled_fixed_spacing = downscaled_fixed.GetSpacing()
    downscaled_moving_spacing = downscaled_moving.GetSpacing()


    downscaled_fixed, downscaled_moving = find_masks(sitk.GetArrayViewFromImage(downscaled_fixed), sitk.GetArrayViewFromImage(downscaled_moving))

    downscaled_fixed = sitk.GetImageFromArray(downscaled_fixed, sitk.sitkFloat64)
    downscaled_moving = sitk.GetImageFromArray(downscaled_moving, sitk.sitkFloat64)
    downscaled_fixed.SetSpacing(downscaled_fixed_spacing)
    downscaled_moving.SetSpacing(downscaled_moving_spacing)


    downscaled_fixed = sitk.VectorMagnitude(downscaled_fixed)
    downscaled_moving = sitk.VectorMagnitude(downscaled_moving)
    downscaled_fixed = sitk.Cast(downscaled_fixed, sitk.sitkFloat64)
    downscaled_moving = sitk.Cast(downscaled_moving, sitk.sitkFloat64)


    plt.title("Images overlapped pre registration (fixed=red, green=moving)")
    plt.show()
    print("####### Starting  Affine registration #######")

    ###################### AFFINE REGISTRATION ########################
    initial_affine = sitk.CenteredTransformInitializer(
        downscaled_fixed,
        downscaled_moving,
        sitk.AffineTransform(downscaled_fixed.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetInitialTransform(initial_affine)
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.001, numberOfIterations=300)
    R.SetMetricSamplingPercentage(0.20)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerScalesFromPhysicalShift()

    affine_transform = R.Execute(downscaled_fixed, downscaled_moving)

    ###################### B-SPLINE REGISTRATION ######################

    transform_domain_mesh_size = [8] * downscaled_fixed.GetDimension()
    bspline_initial = sitk.BSplineTransformInitializer(downscaled_fixed, transform_domain_mesh_size)

    composite_transform = sitk.CompositeTransform([affine_transform, bspline_initial])

    R.SetInitialTransform(composite_transform)

    # reconfigure for B-spline.
    R.SetOptimizerScalesFromJacobian()  # Better for B-splines
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=8.0,
        minStep=1e-5,
        numberOfIterations=400
    )

    print("####### Starting  B-spline registration #######")
    final_composite_transform = R.Execute(downscaled_fixed, downscaled_moving)

    print("Registered")

    print("####### Transforming image #######")

    ###################### FINAL TRANSFORM ########################

    final_image = sitk.Resample(
        moving,
        fixed,
        final_composite_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    return sitk.GetArrayFromImage(final_image)

def run(czi_one, czi_two, output_czi, flip_nissl):
    # preparing metadata for writing
    channel_meta_data_file1 = CziChannelInfo(czi_one)
    channel_meta_data_file2 = CziChannelInfo(czi_two)

    #all_names = channel_meta_data_file1.names + channel_meta_data_file2.names
    display2 = channel_meta_data_file2.czi_disp_settings
    display2[4] = display2.pop(
        0)  # there are 4 channels in file 1, so for each channel in file 2, I have to rename each key.
    all_display = {**channel_meta_data_file1.czi_disp_settings, **display2}

    fixed_image = None


    with czi.create_czi(output_czi, exist_ok=True) as new_czi_file:

        channel_index = 0  # change back
        channel_dict = {}

        print("Combining CZIs")
        print("Copying CZI file one")
        with czi.open_czi(czi_one) as czi_file:  # use last channel in the first file to register to nissl.
            for i in range(len(channel_meta_data_file1.names)):  # for channels/planes in file
                channel_dict[channel_index] = channel_meta_data_file1.names[i].replace('/', '-')
                plane = {'C': i}  # Plane, note i == channel_index
                frame = czi_file.read(plane=plane)
                new_czi_file.write(frame[:, ::-1, :], plane=plane)  # note ::-1 because the image is otherwise rotated.
                channel_index += 1

                if i == len(channel_meta_data_file1.names) - 1:  # saves last possible layer
                    fixed_image = czi_file.read(plane=plane)[:, ::-1, 0]

        print("Copied data in first CZI")
        print("Retrieving Nissl...")
        # Open the second file. We assume this file contains only the NISSL staining.
        # Not too difficult to adapt the code to something more general (say the 2nd file has more channels than just NISSL)
        with czi.open_czi(czi_two) as czi_file:
            # find the channel and save it in nissl_data variable so we can close the czi for more memory
            for i in range(len(channel_meta_data_file2.names)):  # for channels in file
                channel_dict[channel_index] = channel_meta_data_file2.names[i].replace('/', '-')
                # plane = {'C': channel_index}  # Plane

                # if nissl data
                if channel_meta_data_file2.names[i] == "Bright":  # maybe not bright? idk
                    if flip_nissl:
                        nissl_data = czi_file.read(plane={'C': i})[:, ::-1, 0]
                    else:
                        nissl_data = czi_file.read(plane={'C': i})[:, :, 0]

                    nissl_datamin = nissl_data.min()
                    nissl_datamax = nissl_data.max()  # save later for normalization to original values.

        fixed_image = cv2.normalize(fixed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        nissl_data = cv2.normalize(nissl_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        start_time = time.perf_counter()
        frame = register_images(sitk.GetImageFromArray(fixed_image), sitk.GetImageFromArray(nissl_data))


        end_time = time.perf_counter()
        print("Registered and transformed! Time elapsed:", end_time - start_time, "Seconds")

        frame = cv2.normalize(frame, None, alpha=nissl_datamin, beta=nissl_datamax, norm_type=cv2.NORM_MINMAX)  # normalize back to original values.
        frame = frame.astype(np.uint16)

        new_czi_file.write(frame, plane=plane)

        print("Copied CZI file two")
        print("Found and added channels: ", channel_dict)
        new_czi_file.write_metadata(output_czi, channel_names=channel_dict, display_settings=all_display)
        print("New CZI file written at: ", output_czi)





    # TODO

    # - MAKE ABOVE PROCESS A VISUAL EXPERIENCE.
    #   -> GUI to visualize registrations.
    #       -> Choose folder and automatically read files or just select two czis files.
    #       -> Downsized version of the files are displayed, overlapping the last channel of the first and the nissl.
    #       -> Choose to mirror,
    #       -> Choose threshold for mask visually
    #       -> and register, opting for certain parameters and then visualizing the output of the register
    #       -> hit 'next' to save and continue to the next file to be registered.
    # - AUTOMATICALLY FIND IF INPUTS ARE MIRRORED
    #   -> simple rigid registration and metric value comparison with masks.


def main():
    parser = argparse.ArgumentParser(
        description="Takes two similar CZI files, with different channels, and combines them"
                    "Affine alignment is not adapted yet, please pre-align"
                    "Diffeomorphic mapping is not adapted yet"
    )

    parser.add_argument(
        '-i', '--input_dir',
        help="Input folder directory containing nissl and brightfield of seperate files (nissl should be aptly named (i.e. 'filename_NISSL.czi') ",
        required=False
    )

    parser.add_argument(
        '-j', '--czi_one',
        help="Directory for one CZI file to be combined",
        required=False
    )

    parser.add_argument(
        '-k', '--czi_two',
        help="Directory for the second CZI file to be combined",
        required=False
    )

    parser.add_argument(
        '-o', '--output',
        help="Output folder directory for combined file",
        required=True
    )
    parser.add_argument(
        '-m', '--mirrored',
        help="Are all input images mirrored or no? (TO BE UPDATED TO BE AUTOMATIC)",
        required=True
    )


    args = parser.parse_args()

    input_dir = args.input_dir
    czi_one_dir = args.czi_one
    czi_two_dir = args.czi_two
    output_dir = args.output
    flip_nissl = args.mirrored

    if flip_nissl == "True":
        flip_nissl = True
    else:
        flip_nissl = False
    

    non_nissl_paths = []
    nissl_paths = []

    file_exceptions = [] #not czi, no matching nissl find, doesn't match file naming scheme
    exceptions = []
    #note ALL files need similar naming scheme: "RH XYZ..." X is {A,B,C,D,E,F} Y,Z is {0123456789}

    if input_dir:
        # Use a set to keep track of Nissl files already paired to avoid duplicates
        paired_nissl_files = set()

        all_files = []
        for dirpath, _, filenames in os.walk(input_dir):
            for f in filenames:
                all_files.append(os.path.join(dirpath, f))

        for filepath in all_files:
            filename = os.path.basename(filepath)

            # skips processing filenames with nissl/bright/bf etc
            if any(keyword in filename.lower() for keyword in ["nissl", "brightfield", "bf"]):
                continue

            #regex to find patterns like 'RH A01', 'RH A1', or 'LH F12' returns true/false
            match = re.search(r'(RH|LH)\s*[A-F]\d{1,2}', filename)

            if match:
                section_id = match.group(0)#.replace(" ", "")  # Get identifier like "RHA01"

                # --- Find the matching Nissl/Brightfield file ---
                nissl_match_found = False
                for potential_nissl_path in all_files:
                    potential_nissl_name = os.path.basename(potential_nissl_path)

                    # Check if it has the same section ID and is a Nissl/BF file

                    has_id = section_id in potential_nissl_name#.replace(" ", "")
                    is_nissl = any(n in potential_nissl_name.lower() for n in ["nissl", "brightfield", "bf"])

                    if has_id and is_nissl:

                        # Check if this Nissl file has already been paired
                        if potential_nissl_path in paired_nissl_files:
                            file_exceptions.append(filepath)
                            exceptions.append(f"Duplicate match for section {section_id}")
                            nissl_match_found = True  # Mark as found to prevent "no match" error
                            break

                        # Successful pairing
                        non_nissl_paths.append(filepath)
                        nissl_paths.append(potential_nissl_path)
                        paired_nissl_files.add(potential_nissl_path)  # Mark as paired
                        nissl_match_found = True
                        break  # Stop searching once a match is found

                # If the loop finishes without finding a match
                if not nissl_match_found:
                    file_exceptions.append(filepath)
                    exceptions.append(f"No matching Nissl/BF file for section {section_id}")

            else:
                # file doesn't match the required naming scheme (ie 'RH A01')
                file_exceptions.append(filepath)
                exceptions.append("Does not match file naming scheme")

    else:
        print("Error: 'input_dir' is not defined. Please specify the directory to search.")

    print(f"Found {len(nissl_paths)} matching pairs.")
    print(f"Found {len(file_exceptions)} file exceptions.")

    for i in range(len(non_nissl_paths)):
        output_file_name = non_nissl_paths[i].split("\\")[-1].split('.czi')[0]
        output_path = f"{output_dir}/{output_file_name}_combined.czi"
        print(f"Beginning combining {non_nissl_paths[i].split('.czi')[0]} and {nissl_paths[i].split('.czi')[0]} to {output_path}")
        try:
            run(non_nissl_paths[i], nissl_paths[i], output_path, flip_nissl)
        except Exception as e:
            print("Failed to convert file: ", non_nissl_paths[i])
            file_exceptions.append(non_nissl_paths[i])
            exceptions.append(e)
    print("Finished converting files: " + str(len(non_nissl_paths[i])))
    print("Failed files: " + str(len(file_exceptions)))
    for i in range(len(exceptions)):
        print(25 * '_')
        print("File Exceptions: ")
        print(file_exceptions[i])
        print(exceptions[i])
        print(25 * '_')


if __name__ == '__main__':
    main()

