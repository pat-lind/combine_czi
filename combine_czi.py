import argparse
from pylibCZIrw import czi as czi
from czitools.metadata_tools.czi_metadata import CziChannelInfo
from scipy.signal import find_peaks
import numpy as np
import cv2
import time
import SimpleITK as sitk
import os
import re


def get_masks(stack):
    #calculate threshold of which to draw our section mask.
    #based off observation of histograms.
    masks = []

    masks.append(process_mask(stack[0]))

    masks.append(255-process_mask(stack[1])) #nissl, which has different image values so we minus 255


    return masks[0], masks[1]

def calculate_thresh(img):
    histogram, bins = np.histogram(img.ravel(), bins=256, range=(0,256))

    return histo_algo(histogram)


def histo_algo(histogram): #return 2nd peak after first in histogram for the threshold
    peaks, _ = find_peaks(histogram, height=300)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    #print("sorted peaks",sorted_peaks)
    top_two_peaks = sorted_peaks[:2]
    if abs(top_two_peaks[0] - top_two_peaks[1]) > 50: # if their distance is pretty big, then just return the first peak. In our grayscale images there are two big populations of white and black pixels, resembling our two greatest peaks. we want whatever grayscale value of these to be our threshold.
        return top_two_peaks[0]

    other_peaks = [peak for peak in sorted_peaks if peak > top_two_peaks[1]]
    if other_peaks:
        avg_other_peaks = np.mean(other_peaks).astype(int)
        min = np.inf
        for i, num in enumerate(histogram[top_two_peaks[1]:avg_other_peaks]):
            if num < min:
                min = num
            elif num > min:
                return top_two_peaks[1]+i #index of first bump after the 2nd peak in histogram
    else:
        return top_two_peaks[1]

def process_mask(image):
    img = image.copy()
    thresh  = calculate_thresh(image)
    #print(thresh)
    mask = img >= thresh
    img[mask] = 255

    mask = img < thresh
    img[mask] = 0

    img = cv2.GaussianBlur(img, (5, 5), 2)

    return img



def preprocess_stack(stack):
    dapi_mask, nissl_mask  = get_masks(stack)

    return [dapi_mask, nissl_mask]

def register_images(fixedimg,movingimg, mask_stack):

    fixedmask = sitk.Cast(sitk.GetImageFromArray(mask_stack[0]), sitk.sitkFloat32)
    movingmask = sitk.Cast(sitk.GetImageFromArray(mask_stack[1]), sitk.sitkFloat32)



    print("Registering Nissl...")
    print("-" * 30)



    ######################REGISTRATION########################
    initial_affine = sitk.CenteredTransformInitializer(
        fixedmask,
        movingmask,
        sitk.AffineTransform(fixedmask.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetInitialTransform(initial_affine)
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.001, numberOfIterations=150)
    R.SetMetricSamplingPercentage(0.20)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerScalesFromPhysicalShift()

    print("####### Starting Affine Registration #######")
    affine_transform = R.Execute(fixedmask, movingmask)
    print("####### Finished Affine Registration #######")


    ##### B SPLINE #####

    transform_domain_mesh_size = [8] * fixedmask.GetDimension()
    bspline_initial = sitk.BSplineTransformInitializer(fixedmask, transform_domain_mesh_size)

    composite_transform = sitk.CompositeTransform([affine_transform, bspline_initial])

    # By default, the optimizer will ONLY modify the parameters of the LAST transform in the list.
    R.SetInitialTransform(composite_transform)

    # 4. Reconfigure the optimizer for the B-spline stage.
    R.SetOptimizerScalesFromJacobian() # Better for B-splines
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=5.0,
        minStep=1e-5,
        numberOfIterations=320
    )


    print("####### Starting  B-spline Registration #######")
    final_composite_transform = R.Execute(fixedmask, movingmask)
    print("####### Finished B-spline Registration #######")

    print("####### Transforming and Saving Final Image #######")
    ###################### FINAL TRANSFORM ########################
    fixedimg = sitk.Cast(fixedimg, sitk.sitkFloat32) # DAPI
    movingimg = sitk.Cast(movingimg, sitk.sitkFloat32) # NISSL

    # V Final resampled image
    final_image = sitk.Resample(
    movingimg,
    fixedimg,
    final_composite_transform,
    sitk.sitkLinear,
    0.0,
    movingimg.GetPixelID()
    )



    return sitk.GetArrayFromImage(final_image)





def run(czi_one, czi_two, output_czi):
    # preparing metadata for writing
    channel_meta_data_file1 = CziChannelInfo(czi_one)
    channel_meta_data_file2 = CziChannelInfo(czi_two)

    #all_names = channel_meta_data_file1.names + channel_meta_data_file2.names
    display2 = channel_meta_data_file2.czi_disp_settings
    display2[4] = display2.pop(
        0)  # there are 4 channels in file 1, so for each channel in file 2, I have to rename each key.
    all_display = {**channel_meta_data_file1.czi_disp_settings, **display2}

    dapi_data = None


    with czi.create_czi(output_czi, exist_ok=True) as new_czi_file:

        channel_index = 0  # change back
        channel_dict = {}

        print("Combining CZIs")
        print("Copying CZI file one")
        with czi.open_czi(czi_one) as czi_file:
            for i in range(len(channel_meta_data_file1.names)):  # for channels/planes in file
                channel_dict[channel_index] = channel_meta_data_file1.names[i].replace('/', '-')
                plane = {'C': i}  # Plane, note i == channel_index
                frame = czi_file.read(plane=plane)
                new_czi_file.write(frame[:, ::-1, :], plane=plane)  # note ::-1 because the image is otherwise rotated.
                channel_index += 1
                #if all_names[i] == "DAPI":  # Save dapi layer to later register nissl. (Assuming dapi has best data for registration)
                dapi_data = czi_file.read(plane=plane)[:, ::-1, 0] #saves last possible layer
                dapi_data_shape = dapi_data.shape


        if dapi_data is None:
            print("DAPI DATA NOT FOUND IN CZI FILE #1")
        print("Copied CZI file one")

        # Open the second file. We assume this file contains only the NISSL staining.
        # Not too difficult to adapt the code to something more general (say the 2nd file has more channels than just NISSL)
        with czi.open_czi(czi_two) as czi_file:
            for i in range(len(channel_meta_data_file2.names)):  # for channels in file
                channel_dict[channel_index] = channel_meta_data_file2.names[i].replace('/', '-')
                plane = {'C': channel_index}  # Plane

                # if nissl data
                if channel_meta_data_file2.names[i] == "Bright":  # ************
                    nissl_data = czi_file.read(plane={'C': i})[:, :, 0]
                    nissl_data = cv2.resize(nissl_data, (dapi_data_shape[1], dapi_data_shape[0]),
                                            interpolation=cv2.INTER_AREA)  # ensure same size as original dapi data
                    nissl_datamin = nissl_data.min()
                    nissl_datamax = nissl_data.max()  # save later for normalization to original values.



        dapi_data = cv2.normalize(dapi_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        nissl_data = cv2.normalize(nissl_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # register the masks
        start_time = time.perf_counter()
        frame = register_images(sitk.GetImageFromArray(dapi_data), sitk.GetImageFromArray(nissl_data), #we're registering full size masks
                                preprocess_stack(np.stack([dapi_data, nissl_data], axis=0)))
                                # preprocess stack returns a stack of the masks, so we are registering masks of the images, or more accurately, the outlines


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
    #   -> GUI to visualize rotations and registrations.
    #


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


    args = parser.parse_args()

    input_dir = args.input_dir
    czi_one_dir = args.czi_one
    czi_two_dir = args.czi_two
    output_dir = args.output

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
            run(non_nissl_paths[i], nissl_paths[i], output_path)
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
