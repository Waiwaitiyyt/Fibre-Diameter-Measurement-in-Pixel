# Fibre Diameter Measurement in Pixel

Fibre diameter measurement program based on `opencv` and `scipy`. Raw image will be processed and binarized via opencv and skimage, the skeleton of fibre will then be extracted. The diameter measurement is done based on the skeleton, via `distance_transform_edt()` in scipy.

## Usage
Move file `fibreMeasure.py` under the root path of your project, import fibreMeausre module via:

    import fibreMeasure

The default public API for fibreMeasure is `measure()`, it returns the average fibre diameter in pixel for the input image.

    measure(imgPath: str) -> float

## License

This project is released under Apache 2.0 license

## Other Notice

Currently no error handling mechanism developed, so be careful to input the correct `imgPath` into the `measure()` function.



