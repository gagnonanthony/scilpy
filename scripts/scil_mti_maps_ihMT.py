#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes four myelin indices maps from the Magnetization Transfer
(MT) and inhomogeneous Magnetization Transfer (ihMT) images. Magnetization
Transfer is a contrast mechanism in tissue resulting from the proton exchange
between non-aqueous protons (from macromolecules and their closely associated
water molecules, the "bound" pool) and protons in the free water pool called
aqueous protons. This exchange attenuates the MRI signal, introducing
microstructure-dependent contrast. MT's effect reflects the relative density
of macromolecules such as proteins and lipids, it has been associated with
myelin content in white matter of the brain.

Different contrasts can be done with an off-resonance pulse prior to image
acquisition (a prepulse), saturating the protons on non-aqueous molecules,
by applying different frequency irradiation. The two MT maps and two ihMT maps
are obtained using six contrasts: single positive frequency image, single
negative frequency image, dual alternating positive/negative frequency image,
dual alternating negative/positive frequency image (saturated images); 
and two unsaturated contrasts as reference. These two references should be
acquired with predominant PD (proton density) and T1 weighting at different
excitation flip angles (a_PD, a_T1) and repetition times (TR_PD, TR_T1).


Input Data recommendation:
  - it is recommended to use dcm2niix (v1.0.20200331) to convert data
    https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20200331
  - dcm2niix conversion will create all echo files for each contrast and
    corresponding json files
  - all contrasts must have a same number of echoes and coregistered
    between them before running the script
  - Mask must be coregistered to the echo images
  - ANTs can be used for the registration steps (http://stnava.github.io/ANTs/)


The output consists of a ihMT_native_maps folder containing the 4 myelin maps:
    - MTR.nii.gz : Magnetization Transfer Ratio map
    - ihMTR.nii.gz : inhomogeneous Magnetization Transfer Ratio map
    The (ih)MT ratio is a measure reflecting the amount of bound protons.

    - MTsat.nii.gz : Magnetization Transfer saturation map
    - ihMTsat.nii.gz : inhomogeneous Magnetization Transfer saturation map
    The (ih)MT saturation is a pseudo-quantitative maps representing
    the signal change between the bound and free water pools.

As an option, the Complementary_maps folder contains the following images:
    - altnp.nii.gz : dual alternating negative and positive frequency image
    - altpn.nii.gz : dual alternating positive and negative frequency image
    - positive.nii.gz : single positive frequency image
    - negative.nii.gz : single negative frequency image
    - mtoff_PD.nii.gz : unsaturated proton density weighted image
    - mtoff_T1.nii.gz : unsaturated T1 weighted image
    - MTsat_d.nii.gz : MTsat computed from the mean dual frequency images
    - MTsat_sp.nii.gz : MTsat computed from the single positive frequency image
    - MTsat_sn.nii.gz : MTsat computed from the single negative frequency image
    - R1app.nii.gz : Apparent R1 map computed for MTsat.
    - B1_map.nii.gz : B1 map after correction and smoothing (if given).

The final maps from ihMT_native_maps can be corrected for B1+ field
  inhomogeneity, using either an empiric method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method empiric
  or a model-based method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method model_based
  --in_B1_fitValues 3 .mat files, obtained externally from 
    https://github.com/TardifLab/OptimizeIHMTimaging/tree/master/b1Correction,
    and given in this order: positive frequency saturation, negative frequency
    saturation, dual frequency saturation.
For both methods, the nominal value of the B1 map can be set with
  --B1_nominal value


>>> scil_mti_maps_ihMT.py path/to/output/directory
    --in_altnp path/to/echo*altnp.nii.gz --in_altpn path/to/echo*altpn.nii.gz
    --in_mtoff_pd path/to/echo*mtoff.nii.gz --in_negative path/to/echo*neg.nii.gz
    --in_positive path/to/echo*pos.nii.gz --in_mtoff_t1 path/to/echo*T1w.nii.gz
    --mask path/to/mask_bin.nii.gz

By default, the script uses all the echoes available in the input folder.
If you want to use a single echo add --single_echo to the command line and
replace the * with the specific number of the echo.

"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np

from scilpy.io.utils import (get_acq_parameters, add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.io.image import load_img
from scilpy.image.volume_math import concatenate
from scilpy.reconst.mti import (compute_contrasts_maps,
                                compute_ihMT_maps, threshold_maps,
                                apply_B1_correction_empiric,
                                apply_B1_correction_model_based,
                                adjust_b1_map_intensities,
                                smooth_B1_map,
                                read_fit_values_from_mat_files,
                                compute_B1_correction_factor_maps,
                                compute_R1app)

EPILOG = """
Varma G, Girard OM, Prevost VH, Grant AK, Duhamel G, Alsop DC.
Interpretation of magnetization transfer from inhomogeneously broadened lines
(ihMT) in tissues as a dipolar order effect within motion restricted molecules.
Journal of Magnetic Resonance. 1 nov 2015;260:67-76.

Manning AP, Chang KL, MacKay AL, Michal CA. The physical mechanism of
"inhomogeneous" magnetization transfer MRI. Journal of Magnetic Resonance.
1 janv 2017;274:125-36.

Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of
magnetization transfer with inherent correction for RF inhomogeneity
and T1 relaxation obtained from 3D FLASH MRI. Magnetic Resonance in Medicine.
2008;60(6):1396-407.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_dir',
                   help='Path to output folder.')

    p.add_argument('--out_prefix',
                   help='Prefix to be used for each output image.')
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing. '
                        'Not recommended.')
    p.add_argument('--single_echo', action='store_true',
                   help='Use this option when there is only one echo.')
    p.add_argument('--mask',
                   help='Path to the binary brain mask.')
    p.add_argument('--extended_output', action='store_true',
                   help='If set, outputs the folder Complementary_maps.')
    
    b = p.add_argument_group(title='B1 correction')
    b.add_argument('--in_B1_map',
                   help='Path to B1 coregister map to MT contrasts.')
    b.add_argument('--B1_correction_method',
                   choices=['empiric', 'model_based'], default='empiric',
                   help='Choice of B1 correction method. Choose between '
                        'empiric and model-based. Note that the model-based '
                        'method requires a B1 fitvalues file, and will only '
                        'correct the saturation measures. [%(default)s]')
    b.add_argument('--in_B1_fitvalues', nargs=3,
                   help='Path to B1 fitvalues files obtained externally. '
                        'Should be three .mat files given in this specific '
                        'order: positive frequency saturation, negative '
                        'frequency saturation, dual frequency saturation.')
    b.add_argument('--B1_nominal', default=100,
                   help='Nominal value for the B1 map. For Philips, should be '
                        '100. [%(default)s]')

    g = p.add_argument_group(title='ihMT contrasts', description='Path to '
                             'echoes corresponding to contrasts images. All '
                             'constrasts must have the same number of echoes '
                             'and coregistered between them. '
                             'Use * to include all echoes.')
    g.add_argument('--in_altnp', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of Negative and Positive '
                        'frequency saturation pulse.')
    g.add_argument('--in_altpn', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of Positive and Negative '
                        'frequency saturation pulse.')
    g.add_argument("--in_negative", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'Negative frequency saturation pulse.')
    g.add_argument("--in_positive", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'Positive frequency saturation pulse.')
    g.add_argument("--in_mtoff_pd", nargs='+', required=True,
                   help='Path to all echoes corresponding to the predominant '
                        'PD (proton density) weighting images with no '
                        'saturation pulse.')
    g.add_argument("--in_mtoff_t1", nargs='+',
                   help='Path to all echoes corresponding to the predominant '
                        'T1 weighting images with no saturation pulse. This '
                        'one is optional, since it is only needed for the '
                        'calculation of MTsat and ihMTsat. Acquisition '
                        'parameters should also be set with this image.')

    a = p.add_mutually_exclusive_group(title='Acquisition parameters',
                                       required='--in_mtoff_t1' in sys.argv,
                                       help='Acquisition parameters required '
                                            'for MTsat and ihMTsat '
                                            'calculation. These are the '
                                            'excitation flip angles '
                                            '(a_PD, a_T1) and repetition '
                                            'times (TR_PD, TR_T1) of the '
                                            'PD and T1 images.')
    a1 = a.add_argument_group(title='Json files option',
                              help='Use the json files to get the acquisition '
                                   'parameters.')
    a1.add_argument('--in_pd_json',
                   help='Path to MToff PD json file.')
    a1.add_argument('--in_t1_json',
                   help='Path to MToff T1 json file.')
    a2 = a.add_argument_group(title='Parameters values option',
                              help='Give the acquisition parameters directly')
    a2.add_argument('--flip_angles',
                   help='Flip angle of mtoff_PD and mtoff_T1, in that order.')
    a2.add_argument('--rep_times',
                   help='Repetition time of mtoff_PD and mtoff_T1, in that '
                        'order.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.extended_output:
        assert_output_dirs_exist_and_empty(parser, args,
                                           os.path.join(args.out_dir,
                                                        'Complementary_maps'),
                                           os.path.join(args.out_dir,
                                                        'ihMT_native_maps'),
                                           create_dir=True)
    else:
        assert_output_dirs_exist_and_empty(parser, args,
                                           os.path.join(args.out_dir,
                                                        'ihMT_native_maps'),
                                           create_dir=True)

    # Merge all echos path into a list
    maps_list = [args.in_altnp, args.in_altpn, args.in_negative, args.in_positive,
                 args.in_mtoff_pd]
    
    if args.in_mtoff_t1:
        maps_list.append(args.in_mtoff_t1)

    # check echoes number and jsons
    assert_inputs_exist(parser, maps_list)
    for curr_map in maps_list[1:]:
        if len(curr_map) != len(maps_list[0]):
            parser.error('Not the same number of echoes per contrast')

    if args.B1_correction_method == 'model_based' and not args.in_B1_fitvalues:
        parser.error('Fitvalues files must be given when choosing the '
                     'model-based B1 correction method. Please use '
                     '--in_B1_fitvalues.')

    # Set TR and FlipAngle parameters for ihMT (positive contrast)
    # and T1w images
    if args.flip_angles:
        flip_angles = args.flip_angles
        rep_times = args.rep_times
    else:
        for i, curr_json in enumerate(args.in_pd_json, args.in_t1_json):
            acq_parameter = get_acq_parameters(curr_json,
                                               ['RepetitionTime', 'FlipAngle'])
            rep_times[i] = acq_parameter[0] * 1000
            flip_angles[i] = acq_parameter[1] * np.pi / 180.

    # Fix issue from the presence of invalide value and division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Define affine and data shape
    affine = nib.load(maps_list[4][0]).affine
    data_shape = nib.load(maps_list[4][0]).get_fdata().shape

    # Load B1 image
    if args.in_B1_map and args.B1_correction_method == 'model_based':
        B1_img = nib.load(args.in_B1_map)
        B1_map = B1_img.get_fdata(dtype=np.float32)
        B1_map = adjust_b1_map_intensities(B1_map, nominal=args.B1_nominal)
        B1_map = smooth_B1_map(B1_map)
    else:
        B1_map = np.ones(data_shape)

    # Define contrasts maps names
    contrasts_name = ['altnp', 'altpn', 'negative', 'positive', 'mtoff_PD',
                      'mtoff_T1']
    if args.filtering:
        contrasts_name = [curr_name + '_filter'
                          for curr_name in contrasts_name]
    if args.single_echo:
        contrasts_name = [curr_name + '_single_echo'
                          for curr_name in contrasts_name]
    if args.out_prefix:
        contrasts_name = [args.out_prefix + '_' + curr_name
                          for curr_name in contrasts_name]

# Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(maps_list):
        input_images = []
        for image in curr_map:
            img, _ = load_img(image)
            input_images.append(img)
        merged_curr_map = concatenate(input_images, input_images[0])
        computed_contrasts.append(compute_contrasts_maps(
                                  merged_curr_map, filtering=args.filtering,
                                  single_echo=args.single_echo))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float32),
                                 affine),
                 os.path.join(args.out_dir, 'Complementary_maps',
                              contrasts_name[idx] + '.nii.gz'))

    # Compute and thresold ihMT maps
    MTR, ihMTR, MTsat_sp, MTsat_sn, MTsat_d \
        = compute_ihMT_maps(computed_contrasts, parameters, B1_map)

    nib.save(nib.Nifti1Image(MTsat_sp, img.affine), "Complementary_maps/MTsat_sp.nii.gz")
    nib.save(nib.Nifti1Image(MTsat_sn, img.affine), "Complementary_maps/MTsat_sn.nii.gz")
    nib.save(nib.Nifti1Image(MTsat_d, img.affine), "Complementary_maps/MTsat_d.nii.gz")

    if args.in_B1_map and args.B1_correction_method == 'model_based':
        print("Model-based correction")
        cf_eq, r1_to_m0b = read_fit_values_from_mat_files(args.in_B1_fitvalues)
        r1 = compute_R1app(computed_contrasts[2], computed_contrasts[5],
                           parameters, B1_map) * 1000 # convert 1/ms to 1/s
        cf_maps = compute_B1_correction_factor_maps(B1_map, r1, cf_eq,
                                                    r1_to_m0b, b1_ref=1)
        nib.save(nib.Nifti1Image(r1, img.affine), "Complementary_maps/R1obs.nii.gz")
        nib.save(nib.Nifti1Image(np.clip(1/r1, 0, 10), img.affine), "Complementary_maps/T1obs.nii.gz")
        nib.save(nib.Nifti1Image(B1_map, img.affine), "Complementary_maps/B1_map.nii.gz")
        nib.save(nib.Nifti1Image(cf_maps, img.affine), "Complementary_maps/cf_maps.nii.gz")
        MTsat_sp = apply_B1_correction_model_based(MTsat_sp, cf_maps[..., 0])
        MTsat_sn = apply_B1_correction_model_based(MTsat_sn, cf_maps[..., 1])
        MTsat_d = apply_B1_correction_model_based(MTsat_d, cf_maps[..., 2])

    MTsat = (MTsat_sp + MTsat_sn) / 2
    ihMTsat = MTsat_d - MTsat

    if args.in_B1_map and args.B1_correction_method == 'empiric':
        print("Empiric correction")
        B1_img = nib.load(args.in_B1_map)
        B1_map = B1_img.get_fdata(dtype=np.float32)
        B1_map = adjust_b1_map_intensities(B1_map, nominal=args.B1_nominal)
        B1_map = smooth_B1_map(B1_map)
        r1 = compute_R1app(computed_contrasts[2], computed_contrasts[5],
                           parameters, B1_map) * 1000 # convert 1/ms to 1/s
        nib.save(nib.Nifti1Image(r1, img.affine), "Complementary_maps/R1obs.nii.gz")
        nib.save(nib.Nifti1Image(np.clip(1/r1, 0, 10), img.affine), "Complementary_maps/T1obs.nii.gz")
        nib.save(nib.Nifti1Image(B1_map, img.affine), "Complementary_maps/B1_map.nii.gz")
        # MTR = apply_B1_correction_empiric(MTR, B1_map)
        MTsat = apply_B1_correction_empiric(MTsat, B1_map)
        # ihMTR = apply_B1_correction_empiric(ihMTR, B1_map)
        ihMTsat = apply_B1_correction_empiric(ihMTsat, B1_map)

    ihMTR = threshold_maps(ihMTR, args.in_mask, 0, 100,
                           idx_contrast_list=[4, 3, 1, 0, 2],
                           contrasts_maps=computed_contrasts)
    ihMTsat = threshold_maps(ihMTsat, args.in_mask, 0, 10,
                             idx_contrast_list=[4, 3, 1, 0],
                             contrasts_maps=computed_contrasts)      
    MTR = threshold_maps(MTR, args.in_mask, 0, 100,
                         idx_contrast_list=[4, 2],
                         contrasts_maps=computed_contrasts)
    MTsat = threshold_maps(MTsat, args.in_mask, 0, 100,
                           idx_contrast_list=[4, 2],
                           contrasts_maps=computed_contrasts)      

    # Save ihMT and MT images
    img_name = ['ihMTR', 'ihMTsat', 'MTR', 'MTsat']

    if args.filtering:
        img_name = [curr_name + '_filter'
                    for curr_name in img_name]

    if args.single_echo:
        img_name = [curr_name + '_single_echo'
                    for curr_name in img_name]

    if args.in_B1_map:
        img_name = [curr_name + '_B1_corrected'
                    for curr_name in img_name]

    if args.out_prefix:
        img_name = [args.out_prefix + '_' + curr_name
                    for curr_name in img_name]

    img_data = ihMTR, ihMTsat, MTR, MTsat
    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float32),
                                 affine),
                 os.path.join(args.out_dir, 'ihMT_native_maps',
                              name + '.nii.gz'))


if __name__ == '__main__':
    main()
