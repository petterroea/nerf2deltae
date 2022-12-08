import argparse
import json
import os
import imageio
import numpy as np
import cv2
import colour

def load_blender(dataset, testskip, imgdir=None):
    print("Loading from %s" % (imgdir if imgdir is not None else dataset))

    test_file = os.path.join(dataset, "transforms_test.json")
    if not os.path.exists(test_file):
        raise RuntimeError("%s doesn't exist" % test_file)

    with open(test_file, 'r') as f:
        test_images = json.load(f)['frames'][::testskip]
    
    test_images = list(map(lambda image: os.path.join(imgdir if imgdir is not None else dataset, image['file_path']), test_images))

    return test_images

def rgb2boolean(value):
    return (value[0]/3 + value[1]/3 + value[2]/3) > 255/3

rgb2boolean = np.vectorize(rgb2boolean, signature='(3)->()')

def compare_pair(pair):
    nerf_image = cv2.cvtColor(cv2.imread(pair.nerf).astype('float32') / 255.0, cv2.COLOR_RGB2Lab)
    base_truth_image = cv2.cvtColor(cv2.imread(pair.base_truth).astype('float32') / 255.0, cv2.COLOR_RGB2Lab)

    # Flatten the image such that if a mask is enabled we can just slap the mask on
    #nerf_image = nerf_image.reshape(-1, nerf_image.shape[-1])
    #base_truth_image = base_truth_image.reshape(-1, base_truth_image.shape[-1])

    if pair.mask is not None:
        mask = cv2.imread(pair.mask)
        #mask = mask.reshape(-1, mask.shape[-1])
        #print(mask.shape)
        mask_filter = rgb2boolean(mask)

        nerf_image = nerf_image[mask_filter]
        base_truth_image = base_truth_image[mask_filter]

    delta_E = colour.delta_E(nerf_image, base_truth_image)

    return np.mean(delta_E)

class ImagePair():
    def __init__(self, nerf, base_truth, mask):
        self.nerf = nerf
        self.base_truth = base_truth
        self.mask = mask

def main():
    parser = argparse.ArgumentParser(description='Reads a vanilla NeRF dataset, calculates what images are used in the test dataset, and then calculates delta e over those images, given the ground truth. Only works on scenes rendered with render_test. Can optionally use mask_dir to only assess pixels that you want to use')
    parser.add_argument('dataset', type=str, help='Dataset directory for the model')
    parser.add_argument('testset', type=str, help='Output from nerf')
    parser.add_argument('--testskip', type=int, help='Testskip factor, defaults to 8', default=8)
    parser.add_argument('--dataset_type', type=str, choices=['blender'], default="blender", help='Dataset type')
    parser.add_argument('--mask_dir', type=str, help='Object mask directory')

    args = parser.parse_args()

    dataset_images = []
    mask_images = []
    if args.dataset_type == "blender":
        dataset_images = load_blender(args.dataset, args.testskip)

        if args.mask_dir is not None:
            print("Loading masks")
            mask_images = load_blender(args.dataset, args.testskip, args.mask_dir)
    else:
        raise RuntimeError("Invalid dataset_type: \"%s\"" % args.dataset_type)
    
    testset_images = list(filter(lambda file: file.endswith(".png"), os.listdir(args.testset)))
    testset_images.sort()

    if len(testset_images) != len(dataset_images):
        raise RuntimeError("The number of images in %s isn't equal to the test image list! Did you use an incorrect --testskip value?(%s vs %s)" % (args.testset, len(testset_images), len(dataset_images)))

    if args.mask_dir is not None and len(mask_images) != len(dataset_images):
        raise RuntimeError("The number of mask images is not the same as the number of images(%s vs %s)" % (len(mask_images), len(dataset_images)))

    images_combined = zip(testset_images, dataset_images, mask_images) if args.mask_dir is not None else zip(testset_images, dataset_images)

    def class_mapper(item):
        return ImagePair(nerf=os.path.join(args.testset, item[0]), base_truth=("%s.png" % item[1]), mask=("%s.png" % item[2] if len(item)==3 else None))

    pairs = map(class_mapper, images_combined)

    delta_e_values = []
    for pair in pairs:
        deltae = compare_pair(pair)
        delta_e_values.append(deltae)
        print("Delta-E for %s(%s): %f" % (pair.nerf, pair.base_truth, deltae))

    delta_e_values = np.array(delta_e_values)

    print("#" * 19)
    print("# Delta-e report: #")
    print("#" * 19)
    
    print("Min Delta-E: %f" % delta_e_values.min())
    print("Max Delta-E: %f" % delta_e_values.max())
    print("Average Delta-E: %f" % np.mean(delta_e_values))
    print("Standard deviation: %f" % np.std(delta_e_values))



if __name__ == "__main__":
    main()