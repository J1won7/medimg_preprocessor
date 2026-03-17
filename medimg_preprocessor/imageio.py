from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import traceback
from typing import List, Tuple, Type, Union
import warnings

import numpy as np

try:
    import nibabel
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
except ModuleNotFoundError:
    nibabel = None

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    sitk = None

try:
    import tifffile
except ModuleNotFoundError:
    tifffile = None

try:
    from skimage import io as skimage_io
except ModuleNotFoundError:
    skimage_io = None


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _require_dependency(module, package_name: str) -> None:
    if module is None:
        _fail_validation(f"Optional dependency '{package_name}' is required for this reader")


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        if len(input_list) == 1:
            return True
        return np.allclose(input_list[0], input_list[1:])

    @staticmethod
    def _check_all_same_array(input_list):
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        pass


class NibabelIO(BaseReaderWriter):
    supported_file_endings = [
        ".nii",
        ".nii.gz",
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        _require_dependency(nibabel, "nibabel")
        images = []
        original_affines = []
        spacings_for_nnunet = []

        for f in image_fnames:
            nib_image = nibabel.load(f)
            if nib_image.ndim != 3:
                _fail_validation(f"NibabelIO only supports 3D images, got ndim={nib_image.ndim} for file {f}")
            original_affine = nib_image.affine
            original_affines.append(original_affine)
            spacings_for_nnunet.append([float(i) for i in nib_image.header.get_zooms()[::-1]])
            images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            _fail_validation(
                "Not all input images have the same shape. "
                f"Shapes: {[i.shape for i in images]}. Files: {list(image_fnames)}"
            )
        if not self._check_all_same_array(original_affines):
            warnings.warn(
                "Not all input images have the same original affine. "
                "Verify alignment before using the data together.",
                stacklevel=2,
            )
        if not self._check_all_same(spacings_for_nnunet):
            _fail_validation(
                "Not all input images have the same nnU-Net spacing ordering. "
                f"Spacings: {spacings_for_nnunet}. Files: {list(image_fnames)}"
            )

        properties = {
            "nibabel_stuff": {
                "original_affine": original_affines[0],
            },
            "spacing": spacings_for_nnunet[0],
        }
        return np.vstack(images, dtype=np.float32, casting="unsafe"), properties

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        _require_dependency(nibabel, "nibabel")
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)
        seg_nib = nibabel.Nifti1Image(seg, affine=properties["nibabel_stuff"]["original_affine"])
        nibabel.save(seg_nib, output_fname)


class NibabelIOWithReorient(BaseReaderWriter):
    supported_file_endings = [
        ".nii",
        ".nii.gz",
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        _require_dependency(nibabel, "nibabel")
        images = []
        original_affines = []
        reoriented_affines = []
        spacings_for_nnunet = []

        for f in image_fnames:
            nib_image = nibabel.load(f)
            if nib_image.ndim != 3:
                _fail_validation(f"NibabelIOWithReorient only supports 3D images, got ndim={nib_image.ndim} for file {f}")
            original_affine = nib_image.affine
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)
            spacings_for_nnunet.append([float(i) for i in reoriented_image.header.get_zooms()[::-1]])
            images.append(reoriented_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            _fail_validation(
                "Not all input images have the same shape after reorientation. "
                f"Shapes: {[i.shape for i in images]}. Files: {list(image_fnames)}"
            )
        if not self._check_all_same_array(reoriented_affines):
            warnings.warn(
                "Not all input images have the same reoriented affine. "
                "Verify alignment before using the data together.",
                stacklevel=2,
            )
        if not self._check_all_same(spacings_for_nnunet):
            _fail_validation(
                "Not all input images have the same nnU-Net spacing ordering after reorientation. "
                f"Spacings: {spacings_for_nnunet}. Files: {list(image_fnames)}"
            )

        properties = {
            "nibabel_stuff": {
                "original_affine": original_affines[0],
                "reoriented_affine": reoriented_affines[0],
            },
            "spacing": spacings_for_nnunet[0],
        }
        return np.vstack(images, dtype=np.float32, casting="unsafe"), properties

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        _require_dependency(nibabel, "nibabel")
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties["nibabel_stuff"]["reoriented_affine"])
        img_ornt = io_orientation(properties["nibabel_stuff"]["original_affine"])
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
        if not np.allclose(properties["nibabel_stuff"]["original_affine"], seg_nib_reoriented.affine):
            warnings.warn(
                f"Restored affine does not match original affine for file {output_fname}.",
                stacklevel=2,
            )
        nibabel.save(seg_nib_reoriented, output_fname)


def read_nifti_images(
    image_fnames: Union[List[str], Tuple[str, ...]],
    reorient_to_ras: bool = False,
) -> Tuple[np.ndarray, dict]:
    reader = NibabelIOWithReorient() if reorient_to_ras else NibabelIO()
    return reader.read_images(image_fnames)


def read_nifti_seg(seg_fname: str, reorient_to_ras: bool = False) -> Tuple[np.ndarray, dict]:
    reader = NibabelIOWithReorient() if reorient_to_ras else NibabelIO()
    return reader.read_seg(seg_fname)


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".gipl",
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        _require_dependency(sitk, "SimpleITK")
        images = []
        spacings = []
        origins = []
        directions = []
        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")
            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print("ERROR! Not all input images have the same shape!")
            print("Shapes:")
            print([i.shape for i in images])
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print("ERROR! Not all input images have the same spacing!")
            print("Spacings:")
            print(spacings)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print("WARNING! Not all input images have the same origin!")
            print("Origins:")
            print(origins)
            print("Image files:")
            print(image_fnames)
            print("It is up to you to decide whether that's a problem. You should verify overlap manually.")
        if not self._check_all_same(directions):
            print("WARNING! Not all input images have the same direction!")
            print("Directions:")
            print(directions)
            print("Image files:")
            print(image_fnames)
            print("It is up to you to decide whether that's a problem. You should verify overlap manually.")
        if not self._check_all_same(spacings_for_nnunet):
            print("ERROR! Not all input images have the same spacing_for_nnunet!")
            print("spacings_for_nnunet:")
            print(spacings_for_nnunet)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()

        properties = {
            "sitk_stuff": {
                "spacing": spacings[0],
                "origin": origins[0],
                "direction": directions[0],
            },
            "spacing": spacings_for_nnunet[0],
        }
        return np.vstack(images, dtype=np.float32, casting="unsafe"), properties

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        _require_dependency(sitk, "SimpleITK")
        assert seg.ndim == 3, "segmentation must be 3d. If you export a 2d segmentation, provide shape 1,x,y"
        output_dimension = len(properties["sitk_stuff"]["spacing"])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]
        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))
        itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
        itk_image.SetDirection(properties["sitk_stuff"]["direction"])
        sitk.WriteImage(itk_image, output_fname, True)


class SimpleITKIOWithReorient(SimpleITKIO):
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]], orientation="RAS") -> Tuple[np.ndarray, dict]:
        _require_dependency(sitk, "SimpleITK")
        images = []
        spacings = []
        origins = []
        directions = []
        spacings_for_nnunet = []
        original_orientation = None
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_image.GetDirection())
            itk_image = sitk.DICOMOrient(itk_image, orientation)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")
            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print("ERROR! Not all input images have the same shape!")
            print("Shapes:")
            print([i.shape for i in images])
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print("ERROR! Not all input images have the same spacing!")
            print("Spacings:")
            print(spacings)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print("WARNING! Not all input images have the same origin!")
            print("Origins:")
            print(origins)
            print("Image files:")
            print(image_fnames)
            print("It is up to you to decide whether that's a problem. You should verify overlap manually.")
        if not self._check_all_same(directions):
            print("WARNING! Not all input images have the same direction!")
            print("Directions:")
            print(directions)
            print("Image files:")
            print(image_fnames)
            print("It is up to you to decide whether that's a problem. You should verify overlap manually.")
        if not self._check_all_same(spacings_for_nnunet):
            print("ERROR! Not all input images have the same spacing_for_nnunet!")
            print("spacings_for_nnunet:")
            print(spacings_for_nnunet)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()

        properties = {
            "sitk_stuff": {
                "spacing": spacings[0],
                "origin": origins[0],
                "direction": directions[0],
                "original_orientation": original_orientation,
            },
            "spacing": spacings_for_nnunet[0],
        }
        return np.vstack(images, dtype=np.float32, casting="unsafe"), properties

    def write_seg(self, seg, output_fname, properties):
        _require_dependency(sitk, "SimpleITK")
        assert seg.ndim == 3, "segmentation must be 3d. If you export a 2d segmentation, provide shape 1,x,y"
        output_dimension = len(properties["sitk_stuff"]["spacing"])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]
        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8, copy=False))
        itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
        itk_image.SetDirection(properties["sitk_stuff"]["direction"])
        itk_image = sitk.DICOMOrient(itk_image, properties["sitk_stuff"]["original_orientation"])
        sitk.WriteImage(itk_image, output_fname, True)


class Tiff3DIO(BaseReaderWriter):
    supported_file_endings = [
        ".tif",
        ".tiff",
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        _require_dependency(tifffile, "tifffile")
        ending = "." + image_fnames[0].split(".")[-1]
        assert ending.lower() in self.supported_file_endings, f"Ending {ending} not supported by {self.__class__.__name__}"
        ending_length = len(ending)
        truncate_length = ending_length + 5

        images = []
        for f in image_fnames:
            image = tifffile.imread(f)
            if image.ndim != 3:
                raise RuntimeError(f"Only 3D images are supported! File: {f}")
            images.append(image[None])

        expected_aux_file = image_fnames[0][:-truncate_length] + ".json"
        if os.path.isfile(expected_aux_file):
            with open(expected_aux_file, "r", encoding="utf-8") as f:
                spacing = json.load(f)["spacing"]
            assert len(spacing) == 3, f"spacing must have 3 entries. File: {expected_aux_file}"
        else:
            print(f"WARNING no spacing file found for images {image_fnames}\nAssuming spacing (1, 1, 1).")
            spacing = (1, 1, 1)

        if not self._check_all_same([i.shape for i in images]):
            print("ERROR! Not all input images have the same shape!")
            print("Shapes:")
            print([i.shape for i in images])
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images, dtype=np.float32, casting="unsafe"), {"spacing": spacing}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        _require_dependency(tifffile, "tifffile")
        tifffile.imwrite(
            output_fname,
            data=seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False),
            compression="zlib",
        )
        file = os.path.basename(output_fname)
        out_dir = os.path.dirname(output_fname)
        ending = file.split(".")[-1]
        with open(os.path.join(out_dir, file[: -(len(ending) + 1)] + ".json"), "w", encoding="utf-8") as f:
            json.dump({"spacing": properties["spacing"]}, f)

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        _require_dependency(tifffile, "tifffile")
        ending = "." + seg_fname.split(".")[-1]
        assert ending.lower() in self.supported_file_endings, f"Ending {ending} not supported by {self.__class__.__name__}"
        ending_length = len(ending)
        seg = tifffile.imread(seg_fname)
        if seg.ndim != 3:
            raise RuntimeError(f"Only 3D images are supported! File: {seg_fname}")
        seg = seg[None]
        expected_aux_file = seg_fname[:-ending_length] + ".json"
        if os.path.isfile(expected_aux_file):
            with open(expected_aux_file, "r", encoding="utf-8") as f:
                spacing = json.load(f)["spacing"]
            assert len(spacing) == 3, f"spacing must have 3 entries. File: {expected_aux_file}"
            assert all([i > 0 for i in spacing]), f"Spacing must be > 0, spacing: {spacing}"
        else:
            print(f"WARNING no spacing file found for segmentation {seg_fname}\nAssuming spacing (1, 1, 1).")
            spacing = (1, 1, 1)
        return seg.astype(np.float32, copy=False), {"spacing": spacing}


class NaturalImage2DIO(BaseReaderWriter):
    supported_file_endings = [
        ".png",
        ".bmp",
        ".tif",
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        _require_dependency(skimage_io, "scikit-image")
        images = []
        for f in image_fnames:
            npy_img = skimage_io.imread(f)
            if npy_img.ndim == 3:
                assert npy_img.shape[-1] == 3 or npy_img.shape[-1] == 4, (
                    "If image has three dimensions then the last dimension must have shape 3 or 4 "
                    f"(RGB or RGBA). Image shape here is {npy_img.shape}"
                )
                images.append(npy_img.transpose((2, 0, 1))[:, None])
            elif npy_img.ndim == 2:
                images.append(npy_img[None, None])
            else:
                raise RuntimeError(f"Unsupported image ndim {npy_img.ndim} for file {f}")

        if not self._check_all_same([i.shape for i in images]):
            print("ERROR! Not all input images have the same shape!")
            print("Shapes:")
            print([i.shape for i in images])
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images, dtype=np.float32, casting="unsafe"), {"spacing": (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        _require_dependency(skimage_io, "scikit-image")
        skimage_io.imsave(
            output_fname,
            seg[0].astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False),
            check_contrast=False,
        )


LIST_OF_IO_CLASSES = [
    NaturalImage2DIO,
    SimpleITKIO,
    Tiff3DIO,
    NibabelIO,
    NibabelIOWithReorient,
]


def determine_reader_writer_from_file_ending(
    file_ending: str,
    example_file: str = None,
    allow_nonmatching_filename: bool = False,
    verbose: bool = True,
) -> Type[BaseReaderWriter]:
    for rw in LIST_OF_IO_CLASSES:
        if file_ending.lower() in rw.supported_file_endings:
            if example_file is not None:
                try:
                    _ = rw().read_images((example_file,))
                    if verbose:
                        print(f"Using {rw} as reader/writer")
                    return rw
                except Exception:
                    if verbose:
                        print(f"Failed to open file {example_file} with reader {rw}:")
                        traceback.print_exc()
            else:
                if verbose:
                    print(f"Using {rw} as reader/writer")
                return rw
        elif allow_nonmatching_filename and example_file is not None:
            try:
                _ = rw().read_images((example_file,))
                if verbose:
                    print(f"Using {rw} as reader/writer")
                return rw
            except Exception:
                if verbose:
                    print(f"Failed to open file {example_file} with reader {rw}:")
                    traceback.print_exc()
    raise RuntimeError(
        f"Unable to determine a reader for file ending {file_ending} and file {example_file} "
        f"(file None means no file provided)."
    )


def determine_reader_writer_from_dataset_json(
    dataset_json_content: dict,
    example_file: str = None,
    allow_nonmatching_filename: bool = False,
    verbose: bool = True,
) -> Type[BaseReaderWriter]:
    if (
        "overwrite_image_reader_writer" in dataset_json_content
        and dataset_json_content["overwrite_image_reader_writer"] != "None"
    ):
        ioclass_name = dataset_json_content["overwrite_image_reader_writer"]
        for rw in LIST_OF_IO_CLASSES:
            if rw.__name__ == ioclass_name:
                if verbose:
                    print(f"Using {rw} reader/writer")
                return rw
        if verbose:
            print(f"Warning: Unable to find ioclass specified in dataset.json: {ioclass_name}")
            print("Trying to automatically determine desired class")
    return determine_reader_writer_from_file_ending(
        dataset_json_content["file_ending"],
        example_file,
        allow_nonmatching_filename,
        verbose,
    )
